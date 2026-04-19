#!/usr/bin/env python3

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax3, gp_Pln
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_QuasiUniformAbscissa, GCPnts_AbscissaPoint


# ----------------------------
# Data model
# ----------------------------
@dataclass
class CenterSample:
    s: float
    x: float
    y: float
    z: float
    r: float
    rmse: float
    npts: int


# ----------------------------
# Small helpers: paths + interactive
# ----------------------------
def _prompt_str(label: str, default: str = "") -> str:
    d = f" [{default}]" if default else ""
    s = input(f"{label}{d}: ").strip()
    return s if s else default


def _prompt_float(label: str, default: float) -> float:
    while True:
        s = _prompt_str(label, str(default))
        try:
            return float(s)
        except ValueError:
            print("  -> Please enter a number.")


def _prompt_int(label: str, default: int) -> int:
    while True:
        s = _prompt_str(label, str(default))
        try:
            return int(s)
        except ValueError:
            print("  -> Please enter an integer.")


def _prompt_yesno(label: str, default: bool = False) -> bool:
    d = "Y/n" if default else "y/N"
    s = input(f"{label} [{d}]: ").strip().lower()
    if not s:
        return default
    return s in ("y", "yes", "1", "true", "t")


def _resolve_under(base_dir: str, maybe_path: str) -> str:
    maybe_path = (maybe_path or "").strip().strip('"')
    if not maybe_path:
        return maybe_path
    if os.path.isabs(maybe_path):
        return maybe_path

    norm = os.path.normpath(maybe_path)
    base_norm = os.path.normpath(base_dir)

    if norm.startswith(base_norm + os.sep) or norm == base_norm:
        return norm
    return os.path.normpath(os.path.join(base_dir, maybe_path))


def collect_args_interactively(ap: argparse.ArgumentParser) -> argparse.Namespace:
    print("\n=== STEP Centerline Extractor (Interactive Mode) ===")

    # Start from argparse defaults
    ns = ap.parse_args([])
    ns.interactive = True

    print("\n[Paths] (Press Enter to accept defaults)")
    input_step = _prompt_str('STEP file (e.g. "path1.step" or full path)', "path1.step")
    ns.step = _resolve_under(os.path.join("data", "step"), input_step)

    out_name = _prompt_str('Output CSV (e.g. "centerline.csv")', "centerline.csv")
    ns.out = _resolve_under(os.path.join("data", "csv"), out_name)

    out_raw_enable = _prompt_yesno("Also write dense RAW CSV?", default=True)
    ns.out_raw = ""
    if out_raw_enable:
        out_raw_name = _prompt_str('Raw CSV (e.g. "centerline_raw.csv")', "centerline_raw.csv")
        ns.out_raw = _resolve_under(os.path.join("data", "csv"), out_raw_name)

    # AUTO / MANUAL
    print("\n[Parameters]")
    mode = _prompt_str("Parameter mode: auto / manual", "auto").lower()

    if mode in ("auto", "a"):
        preset = _prompt_str("Auto preset: defaults / recommended", "recommended").lower()
        ns = apply_preset(ns, preset)
        ns.no_plot = not _prompt_yesno("Open interactive plot at the end?", default=True)
        print("\nUsing AUTO parameters.")
        return ns

    # MANUAL
    print("\nManual parameter entry (press Enter to accept defaults).")

    print("\n[Bootstrapping]")
    ns.slice_step = _prompt_float("slice_step", ns.slice_step)
    ns.sample_spacing = _prompt_float("sample_spacing", ns.sample_spacing)
    ns.max_rmse = _prompt_float("max_rmse", ns.max_rmse)
    ns.min_points = _prompt_int("min_points", ns.min_points)
    ns.dedupe_eps = _prompt_float("dedupe_eps", ns.dedupe_eps)

    print("\n[Ordering cleanup]")
    ns.outlier_z = _prompt_float("outlier_z", ns.outlier_z)
    ns.knn_k = _prompt_int("knn_k", ns.knn_k)
    ns.max_edge_mult = _prompt_float("max_edge_mult", ns.max_edge_mult)

    print("\n[Refinement]")
    ns.refine_iters = _prompt_int("refine_iters", ns.refine_iters)
    ns.resample_ds = _prompt_float("resample_ds", ns.resample_ds)

    print("\n[Bend detection]")
    ns.angle_thresh_deg = _prompt_float("angle_thresh_deg", ns.angle_thresh_deg)
    ns.bend_spread = _prompt_int("bend_spread", ns.bend_spread)

    print("\n[Final selection]")
    ns.max_points = _prompt_int("max_points", ns.max_points)
    ns.baseline_frac = _prompt_float("baseline_frac", ns.baseline_frac)
    ns.min_idx_gap = _prompt_int("min_idx_gap", ns.min_idx_gap)
    ns.peak_keep = _prompt_int("peak_keep", ns.peak_keep)
    ns.min_spacing = _prompt_float("min_spacing", ns.min_spacing)

    ns.no_plot = not _prompt_yesno("Open interactive plot at the end?", default=True)
    return ns


# ----------------------------
# STEP IO
# ----------------------------
def load_step(step_path: str):
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP: {step_path}")
    reader.TransferRoots()
    return reader.OneShape()


# ----------------------------
# Bounding box
# ----------------------------
def bbox(shape) -> Tuple[np.ndarray, np.ndarray]:
    box = Bnd_Box()
    brepbndlib_Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    bb_min = np.array([xmin, ymin, zmin], dtype=float)
    bb_max = np.array([xmax, ymax, zmax], dtype=float)
    return bb_min, bb_max


def make_plane_frame(axis_dir: np.ndarray, origin: np.ndarray) -> gp_Ax3:
    n = float(np.linalg.norm(axis_dir))
    if n < 1e-12:
        axis_dir = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis_dir = axis_dir / n

    z = gp_Dir(float(axis_dir[0]), float(axis_dir[1]), float(axis_dir[2]))
    o = gp_Pnt(float(origin[0]), float(origin[1]), float(origin[2]))
    return gp_Ax3(o, z)

def bbox_corners(bb_min: np.ndarray, bb_max: np.ndarray) -> np.ndarray:
    x0, y0, z0 = bb_min
    x1, y1, z1 = bb_max
    return np.array(
        [
            [x0, y0, z0],
            [x0, y0, z1],
            [x0, y1, z0],
            [x0, y1, z1],
            [x1, y0, z0],
            [x1, y0, z1],
            [x1, y1, z0],
            [x1, y1, z1],
        ],
        dtype=float,
    )


def axis_sweep_range(bb_min: np.ndarray, bb_max: np.ndarray, axis_dir: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    For a plane with normal=axis_dir, sweep parameter t = dot(axis_unit, p).
    Returns (t_min, t_max, center, axis_unit).
    """
    axis = np.asarray(axis_dir, dtype=float)
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / n

    corners = bbox_corners(bb_min, bb_max)
    proj = corners @ axis
    t_min = float(np.min(proj))
    t_max = float(np.max(proj))
    center = 0.5 * (bb_min + bb_max)
    return t_min, t_max, center, axis


def bootstrap_centers_multi_axis(
    shape,
    bb_min: np.ndarray,
    bb_max: np.ndarray,
    slice_step: float,
    sample_spacing: float,
    min_points: int,
    max_rmse: float,
    use_axes: Optional[List[np.ndarray]] = None,
    dedupe_eps: float = 0.2,
) -> List[CenterSample]:
    """
    Run bootstrap slicing along multiple directions (XYZ + PCA recommended).
    Sweeps correctly for *arbitrary* directions by projecting bbox corners onto axis_dir.
    """
    if use_axes is None:
        use_axes = [
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 1.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 1.0], dtype=float),
        ]

    all_pts = []

    for axis_dir in use_axes:
        t_min, t_max, center, axis_unit = axis_sweep_range(bb_min, bb_max, axis_dir)

        # Sweep planes along axis_unit using t = dot(axis_unit, origin)
        t0 = float(center @ axis_unit)
        t = t_min

        while t <= t_max + 1e-9:
            origin = center + axis_unit * (t - t0)

            ax3 = make_plane_frame(axis_dir=axis_unit, origin=origin)
            plane = gp_Pln(ax3)

            pts3 = section_edges_points(shape, plane, sample_spacing=sample_spacing)
            if pts3.shape[0] < min_points:
                t += slice_step
                continue

            pts2 = project_to_plane(pts3, ax3)
            fit = best_circle_from_section_points(
                pts2,
                min_points=min_points,
                max_rmse=max_rmse,
            )
            if fit is None:
                t += slice_step
                continue

            center2, r, rmse = fit
            if rmse > max_rmse:
                t += slice_step
                continue

            center3 = unproject_from_plane(center2, ax3)
            all_pts.append(center3.astype(float))

            t += slice_step

    if not all_pts:
        return []

    pts = np.asarray(all_pts, dtype=float)
    pts = dedupe_points_eps(pts, eps=float(dedupe_eps))

    return [CenterSample(s=0.0, x=p[0], y=p[1], z=p[2], r=float("nan"), rmse=float("nan"), npts=0) for p in pts]

# ----------------------------
# Section sampling
# ----------------------------
def section_edges_points(shape, plane: gp_Pln, sample_spacing: float) -> np.ndarray:
    sec = BRepAlgoAPI_Section(shape, plane, False)
    sec.Approximation(True)
    sec.Build()
    if not sec.IsDone():
        return np.empty((0, 3), dtype=float)

    pts = []
    exp = TopExp_Explorer(sec.Shape(), TopAbs_EDGE)
    while exp.More():
        edge = exp.Current()
        curve = BRepAdaptor_Curve(edge)
        u0 = curve.FirstParameter()
        u1 = curve.LastParameter()

        try:
            length = float(GCPnts_AbscissaPoint.Length(curve, u0, u1))
        except Exception:
            length = 0.0

        if length <= 1e-9:
            exp.Next()
            continue

        npts = max(2, int(length / max(sample_spacing, 1e-9)) + 1)
        absc = GCPnts_QuasiUniformAbscissa(curve, npts, u0, u1)

        if absc.IsDone() and absc.NbPoints() >= 2:
            for i in range(1, absc.NbPoints() + 1):
                u = absc.Parameter(i)
                p = curve.Value(u)
                pts.append((p.X(), p.Y(), p.Z()))

        exp.Next()

    if not pts:
        return np.empty((0, 3), dtype=float)
    return np.array(pts, dtype=float)


# ----------------------------
# Plane projection utilities
# ----------------------------
def project_to_plane(pts3: np.ndarray, ax3: gp_Ax3) -> np.ndarray:
    o = np.array([ax3.Location().X(), ax3.Location().Y(), ax3.Location().Z()], dtype=float)
    xdir = np.array([ax3.XDirection().X(), ax3.XDirection().Y(), ax3.XDirection().Z()], dtype=float)
    ydir = np.array([ax3.YDirection().X(), ax3.YDirection().Y(), ax3.YDirection().Z()], dtype=float)

    rel = pts3 - o
    u = rel @ xdir
    v = rel @ ydir
    return np.column_stack([u, v])


def unproject_from_plane(center2: np.ndarray, ax3: gp_Ax3) -> np.ndarray:
    o = np.array([ax3.Location().X(), ax3.Location().Y(), ax3.Location().Z()], dtype=float)
    xdir = np.array([ax3.XDirection().X(), ax3.XDirection().Y(), ax3.XDirection().Z()], dtype=float)
    ydir = np.array([ax3.YDirection().X(), ax3.YDirection().Y(), ax3.YDirection().Z()], dtype=float)
    return o + center2[0] * xdir + center2[1] * ydir


# ----------------------------
# Circle fit (Kasa) + fallback
# ----------------------------
def fit_circle_kasa(pts2: np.ndarray) -> Optional[Tuple[np.ndarray, float, float]]:
    if pts2.shape[0] < 10:
        return None

    x = pts2[:, 0]
    y = pts2[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y

    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    D, E, F = sol
    cx = D / 2.0
    cy = E / 2.0
    r2 = F + cx * cx + cy * cy
    if r2 <= 0:
        return None
    r = math.sqrt(float(r2))

    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rmse = float(np.sqrt(np.mean((dist - r) ** 2)))
    return np.array([cx, cy], dtype=float), r, rmse


def section_center_fallback(pts2: np.ndarray) -> Optional[Tuple[np.ndarray, float, float]]:
    if pts2.shape[0] < 6:
        return None
    c = pts2.mean(axis=0)
    d = np.linalg.norm(pts2 - c[None, :], axis=1)
    r = float(np.median(d))
    rmse = float(np.sqrt(np.mean((d - r) ** 2)))
    return c.astype(float), r, rmse


# ----------------------------
# Polyline utilities
# ----------------------------
def polyline_arclength(pts: np.ndarray) -> np.ndarray:
    if pts.shape[0] == 0:
        return np.array([], dtype=float)
    s = np.zeros(pts.shape[0], dtype=float)
    for i in range(1, pts.shape[0]):
        s[i] = s[i - 1] + float(np.linalg.norm(pts[i] - pts[i - 1]))
    return s


def resample_polyline_by_arclength(pts: np.ndarray, ds: float) -> np.ndarray:
    if pts.shape[0] < 2:
        return pts.copy()

    s = polyline_arclength(pts)
    L = float(s[-1])
    if L <= 1e-9:
        return pts[[0]].copy()

    n_new = max(2, int(L / max(ds, 1e-9)) + 1)
    s_new = np.linspace(0.0, L, n_new)

    out = np.zeros((n_new, 3), dtype=float)
    j = 0
    for i, si in enumerate(s_new):
        while j < len(s) - 2 and s[j + 1] < si:
            j += 1
        s0, s1 = s[j], s[j + 1]
        p0, p1 = pts[j], pts[j + 1]
        t = 0.0 if abs(s1 - s0) < 1e-12 else (si - s0) / (s1 - s0)
        out[i] = p0 * (1.0 - t) + p1 * t
    return out


def tangent_at(points: np.ndarray, i: int) -> np.ndarray:
    n = points.shape[0]
    if n < 2:
        return np.array([0.0, 0.0, 1.0], dtype=float)

    if i == 0:
        v = points[1] - points[0]
    elif i == n - 1:
        v = points[n - 1] - points[n - 2]
    else:
        v = points[i + 1] - points[i - 1]

    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return v / norm

# ----------------------------
# Orientation helpers
# ----------------------------
def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v, dtype=float)
    return v / n


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array(
        [
            [0.0, -z,   y],
            [z,    0.0, -x],
            [-y,   x,   0.0],
        ],
        dtype=float,
    )


def _orthogonal_unit(v: np.ndarray) -> np.ndarray:
    v = _normalize(v)
    candidates = [
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
    ]
    a = min(candidates, key=lambda c: abs(float(np.dot(c, v))))
    out = a - np.dot(a, v) * v
    return _normalize(out)


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _normalize(axis)
    K = _skew(axis)
    I = np.eye(3, dtype=float)
    return I + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def rotation_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Minimal rotation that maps unit vector a to unit vector b.
    """
    a = _normalize(a)
    b = _normalize(b)

    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))

    if s < 1e-12:
        if c > 0.0:
            return np.eye(3, dtype=float)
        axis = _orthogonal_unit(a)
        return _rodrigues(axis, math.pi)

    vx = _skew(v)
    I = np.eye(3, dtype=float)
    return I + vx + (vx @ vx) * ((1.0 - c) / (s * s))


def initial_normal_from_up(t0: np.ndarray, world_up: np.ndarray) -> np.ndarray:
    """
    Build an initial normal perpendicular to the first tangent.
    """
    t0 = _normalize(t0)
    up = _normalize(world_up)

    n0 = up - np.dot(up, t0) * t0
    if np.linalg.norm(n0) < 1e-12:
        n0 = _orthogonal_unit(t0)
    return _normalize(n0)


def build_rotation_minimizing_frame(
    tangents: np.ndarray,
    world_up: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=float),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      t_path: tangent
      n_path: normal
      b_path: binormal

    Stable right-handed moving frame along the path.
    """
    n_pts = tangents.shape[0]
    t_path = np.zeros_like(tangents, dtype=float)
    n_path = np.zeros_like(tangents, dtype=float)
    b_path = np.zeros_like(tangents, dtype=float)

    if n_pts == 0:
        return t_path, n_path, b_path

    for i in range(n_pts):
        t_path[i] = _normalize(tangents[i])
        if np.linalg.norm(t_path[i]) < 1e-12:
            t_path[i] = np.array([0.0, 0.0, 1.0], dtype=float)

    n_path[0] = initial_normal_from_up(t_path[0], world_up)
    b_path[0] = _normalize(np.cross(t_path[0], n_path[0]))
    n_path[0] = _normalize(np.cross(b_path[0], t_path[0]))

    for i in range(1, n_pts):
        R = rotation_from_a_to_b(t_path[i - 1], t_path[i])
        n_i = R @ n_path[i - 1]

        n_i = n_i - np.dot(n_i, t_path[i]) * t_path[i]
        if np.linalg.norm(n_i) < 1e-12:
            n_i = initial_normal_from_up(t_path[i], world_up)
        n_i = _normalize(n_i)

        b_i = _normalize(np.cross(t_path[i], n_i))
        n_i = _normalize(np.cross(b_i, t_path[i]))

        n_path[i] = n_i
        b_path[i] = b_i

    return t_path, n_path, b_path


def frame_to_rotmat(
    t: np.ndarray,
    n: np.ndarray,
    b: np.ndarray,
    forward_axis: str = "z",
) -> np.ndarray:
    """
    Build world rotation matrix from path frame.

    forward_axis:
      "z" -> tool local Z follows tangent
      "x" -> tool local X follows tangent
      "y" -> tool local Y follows tangent
    """
    forward_axis = forward_axis.lower()

    if forward_axis == "z":
        x_axis = n
        y_axis = b
        z_axis = t
    elif forward_axis == "x":
        x_axis = t
        y_axis = n
        z_axis = b
    elif forward_axis == "y":
        x_axis = b
        y_axis = t
        z_axis = n
    else:
        raise ValueError("forward_axis must be one of: x, y, z")

    return np.column_stack([
        _normalize(x_axis),
        _normalize(y_axis),
        _normalize(z_axis),
    ])


def rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion [qx, qy, qz, qw].
    """
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    tr = m00 + m11 + m22

    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S

    q = np.array([qx, qy, qz, qw], dtype=float)
    q /= max(np.linalg.norm(q), 1e-12)
    return q


def wrap_deg(a: float) -> float:
    a = (a + 180.0) % 360.0 - 180.0
    if a == -180.0:
        return 180.0
    return a


def rotmat_to_kawasaki_oat_deg(R: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Convert rotation matrix to Kawasaki/Astorino OAT angles in degrees.

    Convention:
        R = Rz(O) @ Ry(A) @ Rz(T)

    Returns:
        np.array([O_deg, A_deg, T_deg], dtype=float)
    """
    r00, r01, r02 = R[0, 0], R[0, 1], R[0, 2]
    r10, r11, r12 = R[1, 0], R[1, 1], R[1, 2]
    r20, r21, r22 = R[2, 0], R[2, 1], R[2, 2]

    A = math.atan2(math.sqrt(max(0.0, r02 * r02 + r12 * r12)), r22)
    sA = math.sin(A)

    if abs(sA) > eps:
        O = math.atan2(r12, r02)
        T = math.atan2(r21, -r20)
    else:
        if r22 > 0.0:
            O = 0.0
            T = math.atan2(r10, r00)
        else:
            O = math.atan2(-r10, r11)
            T = 0.0

    return np.array(
        [
            wrap_deg(math.degrees(O)),
            wrap_deg(math.degrees(A)),
            wrap_deg(math.degrees(T)),
        ],
        dtype=float,
    )

# ----------------------------
# Local refinement (tangent-normal slices)
# ----------------------------
def refine_centers_local_slices(
    shape,
    pts: np.ndarray,
    sample_spacing: float,
    min_points: int,
    max_rmse: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = pts.shape[0]
    new_pts = pts.copy()
    radii = np.full(n, np.nan, dtype=float)
    rmses = np.full(n, np.nan, dtype=float)
    ok = np.zeros(n, dtype=bool)

    for i in range(n):
        t = tangent_at(pts, i)
        ax3 = make_plane_frame(axis_dir=t, origin=pts[i])
        plane = gp_Pln(ax3)

        sec_pts3 = section_edges_points(shape, plane, sample_spacing=sample_spacing)
        if sec_pts3.shape[0] < min_points:
            continue

        sec_pts2 = project_to_plane(sec_pts3, ax3)
        fit = best_circle_from_section_points(
            sec_pts2,
            min_points=min_points,
            max_rmse=max_rmse,
        )
        if fit is None:
            continue

        center2, r, rmse = fit
        if rmse > max_rmse:
            continue

        center3 = unproject_from_plane(center2, ax3)
        new_pts[i] = center3
        radii[i] = r
        rmses[i] = rmse
        ok[i] = True

    return new_pts, radii, rmses, ok


# ----------------------------
# Bend detection + selection
# ----------------------------
def _unit_rows(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n


def smooth_vectors(v: np.ndarray, window: int = 9) -> np.ndarray:
    if window <= 1 or v.shape[0] < window:
        return v
    w = window
    pad = w // 2
    vp = np.pad(v, ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(v, dtype=float)
    for i in range(v.shape[0]):
        out[i] = vp[i : i + w].mean(axis=0)
    return out


def turning_angles_deg(t: np.ndarray) -> np.ndarray:
    t = _unit_rows(t)
    dots = np.sum(t[:-1] * t[1:], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def make_bend_mask(angles_deg: np.ndarray, angle_thresh_deg: float, spread: int = 6) -> np.ndarray:
    n = angles_deg.shape[0] + 1
    bend = np.zeros(n, dtype=bool)

    idx = np.where(angles_deg >= angle_thresh_deg)[0]
    for i in idx:
        bend[i] = True
        bend[i + 1] = True

    if spread > 0:
        bend2 = bend.copy()
        for i in range(n):
            if bend[i]:
                lo = max(0, i - spread)
                hi = min(n, i + spread + 1)
                bend2[lo:hi] = True
        bend = bend2

    return bend


def bend_transition_indices(bend_mask: np.ndarray) -> np.ndarray:
    if bend_mask.size == 0:
        return np.empty((0,), dtype=int)
    changes = np.where(bend_mask[1:] != bend_mask[:-1])[0] + 1
    if changes.size == 0:
        return np.empty((0,), dtype=int)
    extra = np.unique(np.clip(np.r_[changes - 1, changes, changes + 1], 0, bend_mask.size - 1))
    return extra.astype(int)


def strongest_peak_indices(angles_deg: np.ndarray, peak_keep: int) -> np.ndarray:
    if angles_deg.size == 0 or peak_keep <= 0:
        return np.empty((0,), dtype=int)

    candidates = []
    for i in range(1, len(angles_deg) - 1):
        if angles_deg[i] > angles_deg[i - 1] and angles_deg[i] >= angles_deg[i + 1]:
            candidates.append(i)

    if not candidates:
        return np.empty((0,), dtype=int)

    candidates = np.array(candidates, dtype=int)
    mags = angles_deg[candidates]
    order = np.argsort(-mags)
    candidates = candidates[order][:peak_keep]

    return np.unique(np.r_[candidates, candidates + 1]).astype(int)


def uniform_arclength_indices(pts: np.ndarray, m: int) -> np.ndarray:
    n = pts.shape[0]
    if n == 0 or m <= 0:
        return np.empty((0,), dtype=int)
    if m >= n:
        return np.arange(n, dtype=int)

    s = polyline_arclength(pts)
    L = float(s[-1])
    if L <= 1e-12:
        return np.linspace(0, n - 1, m).round().astype(int)

    targets = np.linspace(0.0, L, m)
    idx = [int(np.argmin(np.abs(s - t))) for t in targets]
    return np.unique(np.array(idx, dtype=int))


def budgeted_dp_additions(pts: np.ndarray, fixed: np.ndarray, budget_total: int) -> np.ndarray:
    n = pts.shape[0]
    fixed = np.unique(np.clip(fixed.astype(int), 0, n - 1))
    fixed = np.unique(np.r_[fixed, 0, n - 1])
    fixed.sort()

    if fixed.size >= budget_total:
        return fixed[:budget_total]

    keep = set(int(i) for i in fixed)

    def keep_sorted():
        return np.array(sorted(keep), dtype=int)

    while len(keep) < budget_total:
        ks = keep_sorted()
        best_idx = -1
        best_err = -1.0

        for a_i, b_i in zip(ks[:-1], ks[1:]):
            if b_i <= a_i + 1:
                continue

            a = pts[a_i]
            b = pts[b_i]
            ab = b - a
            denom = float(np.linalg.norm(ab))
            seg = pts[a_i + 1 : b_i]

            if denom < 1e-12:
                dists = np.linalg.norm(seg - a, axis=1)
            else:
                cross = np.cross(seg - a, ab)
                dists = np.linalg.norm(cross, axis=1) / denom

            j = int(np.argmax(dists))
            err = float(dists[j])
            idx = a_i + 1 + j

            if idx not in keep and err > best_err:
                best_err = err
                best_idx = idx

        if best_idx < 0:
            break
        keep.add(int(best_idx))

    return np.array(sorted(keep), dtype=int)


def enforce_min_index_gap(indices: np.ndarray, priority: np.ndarray, min_gap: int) -> np.ndarray:
    if indices.size == 0 or min_gap <= 0:
        return np.unique(indices).astype(int)

    order = np.argsort(-priority)
    chosen = []
    for k in order:
        i = int(indices[k])
        if all(abs(i - j) >= min_gap for j in chosen):
            chosen.append(i)

    return np.array(sorted(set(chosen)), dtype=int)


def select_indices_balanced(
    pts: np.ndarray,
    angles_deg: np.ndarray,
    bend_mask: np.ndarray,
    max_points: int,
    baseline_frac: float,
    peak_keep: int,
    min_idx_gap: int,
) -> np.ndarray:
    n = pts.shape[0]
    if n == 0:
        return np.empty((0,), dtype=int)

    max_points = max(2, int(max_points))
    baseline_frac = float(np.clip(baseline_frac, 0.0, 1.0))

    base_n = max(2, min(max_points, int(round(max_points * baseline_frac))))
    baseline = uniform_arclength_indices(pts, base_n)

    endpoints = np.array([0, n - 1], dtype=int)
    transitions = bend_transition_indices(bend_mask)
    peaks = strongest_peak_indices(angles_deg, peak_keep=peak_keep)
    fixed = np.unique(np.r_[baseline, endpoints, transitions, peaks]).astype(int)
    fixed.sort()

    filled = budgeted_dp_additions(pts, fixed=fixed, budget_total=max_points)
    filled = np.unique(filled).astype(int)

    strength = {}
    for i in filled:
        s = 0.0
        if 0 <= i - 1 < angles_deg.size:
            s = max(s, float(angles_deg[i - 1]))
        if 0 <= i < angles_deg.size:
            s = max(s, float(angles_deg[i]))
        strength[int(i)] = s

    priority = np.zeros(filled.size, dtype=float)
    for k, idx in enumerate(filled):
        s = strength.get(int(idx), 0.0)
        if idx == 0 or idx == n - 1:
            s += 1e6
        if idx in transitions:
            s += 1e5
        if idx in peaks:
            s += 1e4
        if idx in baseline:
            s += 10.0
        priority[k] = s

    gapped = enforce_min_index_gap(filled, priority=priority, min_gap=min_idx_gap)

    if gapped.size < max_points:
        extra = uniform_arclength_indices(pts, max_points)
        gapped = np.unique(np.r_[gapped, extra]).astype(int)

    if gapped.size > max_points:
        pr = np.array(
            [strength.get(int(i), 0.0) + (1e6 if i == 0 or i == n - 1 else 0.0) for i in gapped],
            dtype=float,
        )
        order = np.argsort(-pr)
        gapped = np.array(gapped[order[:max_points]], dtype=int)

    gapped = np.unique(np.r_[gapped, 0, n - 1]).astype(int)
    gapped.sort()
    return gapped


def thin_by_min_arclength(pts: np.ndarray, idx: np.ndarray, min_ds: float) -> np.ndarray:
    if idx.size == 0:
        return idx
    idx = np.unique(idx.astype(int))
    idx.sort()

    s = polyline_arclength(pts)
    keep = [idx[0]]
    last_s = s[idx[0]]

    for i in idx[1:-1]:
        if (s[i] - last_s) >= min_ds:
            keep.append(i)
            last_s = s[i]

    keep.append(idx[-1])
    return np.array(keep, dtype=int)


def top_up_uniform_after_thin(pts: np.ndarray, keep_idx: np.ndarray, max_points: int, min_spacing: float) -> np.ndarray:
    keep_idx = np.unique(keep_idx.astype(int))
    keep_idx.sort()

    if keep_idx.size >= max_points:
        keep_idx = keep_idx[:max_points]
        keep_idx = thin_by_min_arclength(pts, keep_idx, min_ds=min_spacing)
        keep_idx = np.unique(np.r_[keep_idx, 0, len(pts) - 1]).astype(int)
        keep_idx.sort()
        return keep_idx

    extra = uniform_arclength_indices(pts, max_points)
    keep_idx = np.unique(np.r_[keep_idx, extra]).astype(int)
    keep_idx.sort()

    keep_idx = thin_by_min_arclength(pts, keep_idx, min_ds=min_spacing)
    keep_idx = np.unique(np.r_[keep_idx, 0, len(pts) - 1]).astype(int)
    keep_idx.sort()

    if keep_idx.size > max_points:
        keep_idx = keep_idx[:max_points]
        keep_idx = np.unique(np.r_[keep_idx, 0, len(pts) - 1]).astype(int)
        keep_idx.sort()

    return keep_idx


# ----------------------------
# Bootstrap: dedupe + multi-axis + PCA axes
# ----------------------------
def dedupe_points_eps(pts: np.ndarray, eps: float = 0.2) -> np.ndarray:
    if pts.size == 0:
        return pts
    key = np.round(pts / float(eps)).astype(np.int64)
    seen = set()
    out = []
    for i in range(len(pts)):
        k = tuple(key[i])
        if k not in seen:
            seen.add(k)
            out.append(pts[i])
    return np.asarray(out, dtype=float)


def pca_axes_from_points(pts: np.ndarray) -> List[np.ndarray]:
    if pts.shape[0] < 3:
        return [
            np.array([1.0, 0.0, 0.0], float),
            np.array([0.0, 1.0, 0.0], float),
            np.array([0.0, 0.0, 1.0], float),
        ]

    c = pts.mean(axis=0)
    X = pts - c[None, :]
    C = (X.T @ X) / max(1, (len(pts) - 1))
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]

    axes = []
    for i in range(3):
        v = evecs[:, i]
        n = float(np.linalg.norm(v))
        if n > 1e-12:
            axes.append((v / n).astype(float))
    return axes


# ----------------------------
# Ordering: outlier filter + mutual-kNN component + geodesic sort
# ----------------------------
def knn_median_distance(pts: np.ndarray, k: int = 8) -> np.ndarray:
    n = len(pts)
    if n <= 1:
        return np.zeros(n, dtype=float)
    dmat = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(dmat, np.inf)
    kk = min(max(1, k), n - 1)
    nn = np.partition(dmat, kth=kk - 1, axis=1)[:, :kk]
    return np.median(nn, axis=1)


def filter_outliers_knn(pts: np.ndarray, k: int = 8, z: float = 3.5) -> np.ndarray:
    if len(pts) < 10:
        return pts
    md = knn_median_distance(pts, k=k)
    med = float(np.median(md))
    mad = float(np.median(np.abs(md - med))) + 1e-12
    rz = 0.6745 * (md - med) / mad
    keep = rz < z
    return pts[keep]


def build_mutual_knn_graph(pts: np.ndarray, k: int = 10, max_edge_mult: float = 3.0):
    n = len(pts)
    dmat = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(dmat, np.inf)
    kk = min(max(2, k), n - 1)

    md = knn_median_distance(pts, k=min(8, kk))
    scale = float(np.median(md)) + 1e-12
    max_edge = max_edge_mult * scale

    knn = []
    for i in range(n):
        nn = np.argpartition(dmat[i], kth=kk - 1)[:kk]
        nn = [int(j) for j in nn if dmat[i, j] <= max_edge]
        knn.append(set(nn))

    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in knn[i]:
            if i in knn[j]:
                w = float(dmat[i, j])
                adj[i].append((j, w))
    return adj


def largest_connected_component(adj):
    n = len(adj)
    seen = np.zeros(n, dtype=bool)
    best = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = [i]
        while stack:
            u = stack.pop()
            for v, _ in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
                    comp.append(v)
        if len(comp) > len(best):
            best = comp
    return np.array(best, dtype=int)


def dijkstra_dense(adj, src: int) -> np.ndarray:
    n = len(adj)
    dist = np.full(n, np.inf, dtype=float)
    dist[src] = 0.0
    visited = np.zeros(n, dtype=bool)
    for _ in range(n):
        u = int(np.argmin(np.where(visited, np.inf, dist)))
        if not np.isfinite(dist[u]):
            break
        visited[u] = True
        for v, w in adj[u]:
            if not visited[v]:
                nd = dist[u] + w
                if nd < dist[v]:
                    dist[v] = nd
    return dist


def order_points_component_geodesic(pts: np.ndarray, k: int = 10, max_edge_mult: float = 3.0) -> np.ndarray:
    if len(pts) <= 1:
        return np.arange(len(pts), dtype=int)

    adj = build_mutual_knn_graph(pts, k=k, max_edge_mult=max_edge_mult)
    comp = largest_connected_component(adj)
    if comp.size < 2:
        axis = int(np.argmax(np.ptp(pts, axis=0)))
        return np.argsort(pts[:, axis])

    idx_map = {int(old): new for new, old in enumerate(comp)}
    pts_c = pts[comp]
    adj_c = [[] for _ in range(len(comp))]
    for old_u in comp:
        u = idx_map[int(old_u)]
        for old_v, w in adj[int(old_u)]:
            if int(old_v) in idx_map:
                v = idx_map[int(old_v)]
                adj_c[u].append((v, w))

    d0 = dijkstra_dense(adj_c, 0)
    a = int(np.nanargmax(np.where(np.isfinite(d0), d0, -1.0)))
    da = dijkstra_dense(adj_c, a)

    if not np.all(np.isfinite(da)):
        axis = int(np.argmax(np.ptp(pts_c, axis=0)))
        return comp[np.argsort(pts_c[:, axis])]

    order_c = np.argsort(da)
    return comp[order_c]


# ----------------------------
# Plot helper (optional)
# ----------------------------
def write_interactive_plot(csv_path: str, html_path: str):
    try:
        import pandas as pd
        import plotly.graph_objects as go
        import webbrowser
    except Exception as e:
        print(f"[plot] Skipping interactive plot (missing deps): {e}")
        return

    df = pd.read_csv(csv_path)

    # Always sort by path order
    if "i" in df.columns:
        df = df.sort_values("i")

    fig = go.Figure()

    # 1) one continuous polyline (NO color split)
    fig.add_trace(
        go.Scatter3d(
            x=df["x"], y=df["y"], z=df["z"],
            mode="lines",
            name="centerline"
        )
    )

    # 2) markers colored by bend (optional)
    if "bend" in df.columns:
        fig.add_trace(
            go.Scatter3d(
                x=df["x"], y=df["y"], z=df["z"],
                mode="markers",
                name="points",
                marker=dict(
                    size=4,
                    color=df["bend"],  # 0/1
                    colorscale="RdBu",
                    opacity=0.9
                )
            )
        )
    else:
        fig.add_trace(
            go.Scatter3d(
                x=df["x"], y=df["y"], z=df["z"],
                mode="markers",
                name="points",
                marker=dict(size=4, opacity=0.9)
            )
        )

    fig.update_layout(
        title="Centerline (selected)",
        scene_aspectmode="data",
        scene_camera=dict(
            projection=dict(type="orthographic"),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=3.0, y=3.0, z=3.0),
        ),
    )

    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    fig.write_html(html_path, auto_open=False)
    webbrowser.open(f"file:///{os.path.abspath(html_path)}")
    print(f"[plot] Opened: {html_path}")

def apply_preset(ns: argparse.Namespace, preset: str) -> argparse.Namespace:
    preset = (preset or "").lower().strip()
    if preset in ("defaults", "default", ""):
        return ns

    if preset in ("recommended", "rec", "r"):
        ns.slice_step = 1.0
        ns.sample_spacing = 0.25
        ns.max_rmse = 5.0
        ns.min_points = 6

        ns.refine_iters = 3
        ns.resample_ds = 2.0
        ns.angle_thresh_deg = 0.5
        ns.bend_spread = 6
        ns.max_points = 100
        ns.baseline_frac = 0.65
        ns.min_idx_gap = 2
        ns.peak_keep = 15
        ns.min_spacing = 10.0

        # cleanup/order knobs
        ns.dedupe_eps = 0.2
        ns.outlier_z = 6.0          # <- less aggressive (keeps end legs)
        ns.knn_k = 10
        ns.max_edge_mult = 3.0
        return ns

    print(f'Unknown preset "{preset}", using defaults.')
    return ns

def best_circle_from_section_points(
    pts2: np.ndarray,
    min_points: int,
    max_rmse: float,
    cluster_eps: Optional[float] = None,
) -> Optional[Tuple[np.ndarray, float, float]]:
    """
    Section may contain multiple loops. Cluster in 2D, fit circle per cluster,
    return the best (lowest RMSE) valid fit.
    """
    if pts2.shape[0] < min_points:
        return None

    # choose a scale if not given
    if cluster_eps is None:
        # typical neighbor distance in 2D (robust)
        d = np.linalg.norm(pts2[:, None, :] - pts2[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        k = min(8, pts2.shape[0] - 1)
        nn = np.partition(d, kth=k - 1, axis=1)[:, :k]
        cluster_eps = float(np.median(nn)) * 3.0 + 1e-9  # generous within-loop connectivity

    n = pts2.shape[0]
    used = np.zeros(n, dtype=bool)
    clusters = []

    # BFS clustering using distance threshold
    for i in range(n):
        if used[i]:
            continue
        queue = [i]
        used[i] = True
        comp = [i]
        while queue:
            u = queue.pop()
            # neighbors within eps
            du = np.linalg.norm(pts2 - pts2[u], axis=1)
            nbrs = np.where((~used) & (du <= cluster_eps))[0]
            for v in nbrs.tolist():
                used[v] = True
                queue.append(v)
                comp.append(v)
        clusters.append(np.array(comp, dtype=int))

    best = None
    best_rmse = float("inf")

    for idx in clusters:
        if idx.size < min_points:
            continue
        fit = fit_circle_kasa(pts2[idx])
        if fit is None:
            fit = section_center_fallback(pts2[idx])
        if fit is None:
            continue
        c2, r, rmse = fit
        if rmse <= max_rmse and rmse < best_rmse:
            best = (c2, r, rmse)
            best_rmse = rmse

    return best

def remove_big_jumps(pts: np.ndarray, z: float = 6.0) -> np.ndarray:
    """
    Remove points that create abnormally large segment lengths.
    Keeps endpoints.
    """
    if len(pts) < 5:
        return pts

    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    med = float(np.median(seg))
    mad = float(np.median(np.abs(seg - med))) + 1e-12
    rz = 0.6745 * (seg - med) / mad

    keep = np.ones(len(pts), dtype=bool)
    bad = np.where(rz > z)[0]
    for i in bad:
        j = i + 1
        if 0 < j < len(pts) - 1:
            keep[j] = False

    pts2 = pts[keep]
    if len(pts2) >= 2:
        pts2[0] = pts[0]
        pts2[-1] = pts[-1]
    return pts2

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="STEP -> centerline (multi-axis + PCA bootstrap, robust ordering, local refinement, <=N point selection)."
    )

    ap.add_argument("--interactive", action="store_true", help="Prompt for inputs interactively")

    ap.add_argument("--step", required=False, help="Input STEP file path (.stp/.step)")
    ap.add_argument("--out", required=False, help="Output CSV path")
    ap.add_argument("--out_raw", default="", help="Optional dense CSV (all refined stations)")
    ap.add_argument("--no_plot", action="store_true", help="Disable interactive plot output")

    # Bootstrap + fit
    ap.add_argument("--slice_step", type=float, default=2.0)
    ap.add_argument("--sample_spacing", type=float, default=0.5)
    ap.add_argument("--max_rmse", type=float, default=2.0)
    ap.add_argument("--min_points", type=int, default=10)

    # Bootstrap cleanup / ordering controls
    ap.add_argument("--dedupe_eps", type=float, default=0.2)
    ap.add_argument("--outlier_z", type=float, default=3.5)
    ap.add_argument("--knn_k", type=int, default=10)
    ap.add_argument("--max_edge_mult", type=float, default=3.0)

    # Refinement
    ap.add_argument("--refine_iters", type=int, default=3)
    ap.add_argument("--resample_ds", type=float, default=2.0)

    # Bend detection + selection
    ap.add_argument("--angle_thresh_deg", type=float, default=0.5)
    ap.add_argument("--bend_spread", type=int, default=6)
    ap.add_argument("--max_points", type=int, default=100)
    ap.add_argument("--baseline_frac", type=float, default=0.65)
    ap.add_argument("--min_idx_gap", type=int, default=2)
    ap.add_argument("--peak_keep", type=int, default=15)
    ap.add_argument("--min_spacing", type=float, default=10.0)

    args = ap.parse_args()

    if args.interactive or not args.step or not args.out:
        args = collect_args_interactively(ap)

    args.step = _resolve_under(os.path.join("data", "step"), args.step)
    args.out = _resolve_under(os.path.join("data", "csv"), args.out)
    if args.out_raw:
        args.out_raw = _resolve_under(os.path.join("data", "csv"), args.out_raw)

    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.out_raw and os.path.dirname(args.out_raw):
        os.makedirs(os.path.dirname(args.out_raw), exist_ok=True)

    print("[1/7] Loading STEP file...", flush=True)
    shape = load_step(args.step)

    print("[2/7] Computing bounding box...", flush=True)
    bb_min, bb_max = bbox(shape)
    print(f"      bbox min={bb_min}, max={bb_max}", flush=True)

    # Seed bootstrap (XYZ) -> PCA axes
    print("[3/7] Bootstrapping centers (XYZ seed -> PCA -> XYZ+PCA)...", flush=True)
    xyz_axes = [
        np.array([1.0, 0.0, 0.0], float),
        np.array([0.0, 1.0, 0.0], float),
        np.array([0.0, 0.0, 1.0], float),
    ]
    seed = bootstrap_centers_multi_axis(
        shape=shape,
        bb_min=bb_min,
        bb_max=bb_max,
        slice_step=args.slice_step,
        sample_spacing=args.sample_spacing,
        min_points=args.min_points,
        max_rmse=args.max_rmse,
        use_axes=xyz_axes,
        dedupe_eps=float(args.dedupe_eps),
    )
    if not seed:
        raise RuntimeError("No valid bootstrap slices found (seed). Try larger slice_step / sample_spacing / max_rmse.")

    pts_seed = np.array([[c.x, c.y, c.z] for c in seed], dtype=float)
    pca_axes = pca_axes_from_points(pts_seed)

    use_axes = xyz_axes + pca_axes
    centers = bootstrap_centers_multi_axis(
        shape=shape,
        bb_min=bb_min,
        bb_max=bb_max,
        slice_step=args.slice_step,
        sample_spacing=args.sample_spacing,
        min_points=args.min_points,
        max_rmse=args.max_rmse,
        use_axes=use_axes,
        dedupe_eps=float(args.dedupe_eps),
    )
    print(f"      bootstrap centers (raw): {len(centers)}", flush=True)
    if not centers:
        raise RuntimeError("No valid bootstrap slices found.")

    # Robust cleanup + ordering
    print("[4/7] Cleaning + ordering + resampling...", flush=True)
    pts0 = np.array([[c.x, c.y, c.z] for c in centers], dtype=float)

    before = len(pts0)
    pts0 = filter_outliers_knn(pts0, k=int(args.knn_k), z=float(args.outlier_z))
    after = len(pts0)
    print(f"      outlier filter: {before} -> {after}", flush=True)

    if len(pts0) < 4:
        raise RuntimeError("Too few bootstrap points after outlier filtering. Reduce outlier_z or dedupe_eps.")

    # Order points by path connectivity, not by one PCA projection
    order = order_points_component_geodesic(
        pts0,
        k=int(args.knn_k),
        max_edge_mult=float(args.max_edge_mult),
    )
    pts0 = pts0[order]

    # Optional: choose direction consistently
    if pts0.shape[0] >= 2 and pts0[0, 0] > pts0[-1, 0]:
        pts0 = pts0[::-1]

    # Remove any remaining pathological long jumps
    pts0 = remove_big_jumps(pts0, z=6.0)

    pts = resample_polyline_by_arclength(pts0, ds=max(1e-6, float(args.resample_ds)))
    print(f"      resampled stations: {len(pts)}", flush=True)

    # Local refinement
    print("[5/7] Refining centerline (local slicing)...", flush=True)
    radii = np.full(len(pts), np.nan, dtype=float)
    rmses = np.full(len(pts), np.nan, dtype=float)

    for it in range(max(0, int(args.refine_iters))):
        print(f"      refinement iteration {it+1}/{args.refine_iters}", flush=True)
        pts_new, r_new, e_new, ok = refine_centers_local_slices(
            shape=shape,
            pts=pts,
            sample_spacing=args.sample_spacing,
            min_points=args.min_points,
            max_rmse=args.max_rmse,
        )
        pts = pts_new
        radii[ok] = r_new[ok]
        rmses[ok] = e_new[ok]

            # Re-order once more after refinement, then clean and resample lightly
    order = order_points_component_geodesic(
        pts,
        k=int(args.knn_k),
        max_edge_mult=float(args.max_edge_mult),
    )
    pts = pts[order]
    radii = radii[order]
    rmses = rmses[order]

    if pts.shape[0] >= 2 and pts[0, 0] > pts[-1, 0]:
        pts = pts[::-1]
        radii = radii[::-1]
        rmses = rmses[::-1]

    pts = remove_big_jumps(pts, z=6.0)

    # Tangents + orientations + bend detection
    print("[6/7] Detecting bends + building orientations...", flush=True)
    tangents = np.zeros_like(pts, dtype=float)
    for i in range(len(pts)):
        tangents[i] = tangent_at(pts, i)

    t_smooth = smooth_vectors(tangents, window=9)

    tool_forward_axis = "x"   # or "z" if you switch back
    tool_roll_deg = -90.0     # adjust if needed

    t_path, n_path, b_path = build_rotation_minimizing_frame(
        t_smooth,
        world_up=np.array([0.0, 0.0, 1.0], dtype=float),
    )

    quats = np.zeros((len(pts), 4), dtype=float)
    oats = np.zeros((len(pts), 3), dtype=float)

    for i in range(len(pts)):
        R = frame_to_rotmat(
            t=t_path[i],
            n=n_path[i],
            b=b_path[i],
            forward_axis=tool_forward_axis,
        )

        # fixed roll around the tool's own forward axis
        if tool_forward_axis == "x":
            R = R @ _rodrigues(
                np.array([1.0, 0.0, 0.0], dtype=float),
                math.radians(tool_roll_deg)
            )
        elif tool_forward_axis == "y":
            R = R @ _rodrigues(
                np.array([0.0, 1.0, 0.0], dtype=float),
                math.radians(tool_roll_deg)
            )
        elif tool_forward_axis == "z":
            R = R @ _rodrigues(
                np.array([0.0, 0.0, 1.0], dtype=float),
                math.radians(tool_roll_deg)
            )

        quats[i] = rotmat_to_quat_xyzw(R)
        oats[i] = rotmat_to_kawasaki_oat_deg(R)

    # Keep quaternion sign continuous
    for i in range(1, len(quats)):
        if float(np.dot(quats[i - 1], quats[i])) < 0.0:
            quats[i] = -quats[i]

    # Safety check
    qnorms = np.linalg.norm(quats, axis=1)
    bad = np.where(qnorms < 0.5)[0]
    if len(bad) > 0:
        raise RuntimeError("Quaternion generation failed: some rows are near zero.")

    angles = turning_angles_deg(t_smooth)
    bend_mask = make_bend_mask(angles, args.angle_thresh_deg, spread=args.bend_spread)

    # Final selection
    print("[7/7] Selecting final points + writing CSV...", flush=True)
    keep_idx = select_indices_balanced(
        pts=pts,
        angles_deg=angles,
        bend_mask=bend_mask,
        max_points=args.max_points,
        baseline_frac=args.baseline_frac,
        peak_keep=args.peak_keep,
        min_idx_gap=args.min_idx_gap,
    )

    keep_idx = thin_by_min_arclength(pts, keep_idx, min_ds=float(args.min_spacing))
    keep_idx = top_up_uniform_after_thin(
        pts=pts,
        keep_idx=keep_idx,
        max_points=int(args.max_points),
        min_spacing=float(args.min_spacing),
    )
    keep_idx.sort()


    if args.out_raw:
        with open(args.out_raw, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "i",
                "x", "y", "z",
                "qx", "qy", "qz", "qw",
                "o", "a", "t",
                "radius", "rmse", "bend"
            ])
            for i in range(len(pts)):
                qx, qy, qz, qw = quats[i]
                o_deg, a_deg, t_deg = oats[i]

                w.writerow([
                    i,
                    pts[i, 0],
                    pts[i, 1],
                    pts[i, 2],
                    qx, qy, qz, qw,
                    o_deg, a_deg, t_deg,
                    radii[i] if np.isfinite(radii[i]) else "",
                    rmses[i] if np.isfinite(rmses[i]) else "",
                    int(bend_mask[i]),
                ])

    with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "i",
                "x", "y", "z",
                "qx", "qy", "qz", "qw",
                "o", "a", "t",
                "radius", "rmse", "bend"
            ])
            for i in keep_idx:
                qx, qy, qz, qw = quats[i]
                o_deg, a_deg, t_deg = oats[i]

                w.writerow([
                    i,
                    pts[i, 0],
                    pts[i, 1],
                    pts[i, 2],
                    qx, qy, qz, qw,
                    o_deg, a_deg, t_deg,
                    radii[i] if np.isfinite(radii[i]) else "",
                    rmses[i] if np.isfinite(rmses[i]) else "",
                    int(bend_mask[i]),
                ])

    print(f"Done | bootstrap={len(centers)} refined={len(pts)} exported={len(keep_idx)}", flush=True)
    print(f"Output CSV: {args.out}", flush=True)
    if args.out_raw:
        print(f"Raw CSV:    {args.out_raw}", flush=True)

    if not args.no_plot:
        plot_path = os.path.join("data", "plots", "centerline.html")
        write_interactive_plot(args.out, plot_path)

if __name__ == "__main__":
    main()