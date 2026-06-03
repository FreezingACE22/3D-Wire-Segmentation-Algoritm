#!/usr/bin/env python3
r"""
Convert CSV path points into an ASTORINO .lc TRANS point file.

This version automatically handles the usual project structure:

    3D-Wire-Segmentation-Algoritm/
    ├── src/
    │   └── csv_to_lc_converter.py
    └── data/
        └── csv/
            └── path7.csv

You can run it from the project root, from src, from data/csv, or from VS Code.

Common commands:

    # Show a list of CSV files in data/csv, ask which one to convert,
    # write same-name .lc beside the selected CSV, place the first point at 0,0,0,
    # use fixed O,A,T = 0,0,0, and name points P1, P2, ... by default.
    python .\src\csv_to_lc_converter.py

    # Give only the CSV file name; script finds it in data/csv automatically
    # and writes path7.lc beside path7.csv. .csv may be omitted.
    python .\src\csv_to_lc_converter.py path7

    # Optional: give a custom LC output name
    python .\src\csv_to_lc_converter.py path7.csv custom_name.lc

    # Optional: override the default first-point position and orientation
    python .\src\csv_to_lc_converter.py path7 --align-first-to 100 200 300 --fixed-oat 0 0 90

    # Optional: use O,A,T values from the CSV instead of fixed 0,0,0
    python .\src\csv_to_lc_converter.py path7 --use-csv-oat

    # Convert all CSV files in data/csv, each with its first point at 0,0,0
    python .\src\csv_to_lc_converter.py --all

Expected CSV columns:
    x, y, z, o, a, t

Output .lc format:
    .JOINTS
    .END
    .TRANS
    P1      X    Y    Z    O    A    T    0.000    1.0    ;
    P2      X    Y    Z    O    A    T    0.000    1.0    ;
    .END
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


# ----------------------------
# Data models
# ----------------------------
@dataclass
class CsvPoint:
    x: float
    y: float
    z: float
    o: float
    a: float
    t: float


@dataclass
class RobotPoint:
    name: str
    x: float
    y: float
    z: float
    o: float
    a: float
    t: float


# ----------------------------
# Automatic project path handling
# ----------------------------
def find_project_root() -> Path:
    """
    Find the project root by searching upward from this script and current folder.
    The root is detected by the presence of data/csv.
    """
    candidates: List[Path] = []

    try:
        candidates.append(Path(__file__).resolve().parent)
    except NameError:
        pass

    candidates.append(Path.cwd().resolve())

    checked: set[Path] = set()
    for start in candidates:
        for folder in [start, *start.parents]:
            if folder in checked:
                continue
            checked.add(folder)
            if (folder / "data" / "csv").is_dir():
                return folder

    # Fallback: if script is in src, parent is probably project root.
    script_dir = Path(__file__).resolve().parent
    if script_dir.name.lower() == "src":
        return script_dir.parent
    return Path.cwd().resolve()


def default_csv_dir(project_root: Path) -> Path:
    return project_root / "data" / "csv"


def newest_csv(csv_dir: Path) -> Path:
    files = sorted(csv_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No .csv files found in {csv_dir}")
    return files[0]


def choose_csv_file_interactive(csv_dir: Path) -> Path:
    """
    Show the user all CSV files in data/csv and ask which one to convert.
    The user may enter either the number from the list or a file name.
    """
    if not csv_dir.is_dir():
        raise FileNotFoundError(f"CSV folder not found: {csv_dir}")

    csv_files = sorted(csv_dir.glob("*.csv"), key=lambda p: p.name.lower())
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found in {csv_dir}")

    print(f"\nCSV files found in: {csv_dir}")
    for i, path in enumerate(csv_files, start=1):
        print(f"  {i}. {path.name}")

    while True:
        choice = input("\nEnter file number or CSV file name to convert: ").strip().strip('"').strip("'")

        if not choice:
            print("Please enter a number or file name.")
            continue

        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(csv_files):
                return csv_files[index - 1].resolve()
            print(f"Number must be between 1 and {len(csv_files)}.")
            continue

        raw = Path(choice)

        # Full or relative path typed manually.
        if raw.exists():
            return raw.resolve()

        # File name typed manually. Allow both path7 and path7.csv.
        possible_names = [raw.name]
        if raw.suffix.lower() != ".csv":
            possible_names.append(raw.name + ".csv")

        for name in possible_names:
            candidate = csv_dir / name
            if candidate.exists():
                return candidate.resolve()

        print(f"Could not find '{choice}' in {csv_dir}. Try again.")


def resolve_csv_path(csv_arg: Optional[str], project_root: Path) -> Path:
    """
    Resolve input CSV from:
    - interactive selection from project data/csv when omitted
    - explicit full/relative path
    - filename in current directory
    - filename in project data/csv

    If the user writes path7 instead of path7.csv, .csv is added automatically
    when looking inside data/csv and the project root.
    """
    csv_dir = default_csv_dir(project_root)

    if not csv_arg:
        return choose_csv_file_interactive(csv_dir)

    raw = Path(csv_arg)

    # Full or relative path that exists.
    if raw.exists():
        return raw.resolve()

    possible_raw_paths = [raw]
    if raw.suffix.lower() != ".csv":
        possible_raw_paths.append(raw.with_suffix(".csv"))

    # Try relative to current folder.
    for candidate in possible_raw_paths:
        if candidate.exists():
            return candidate.resolve()

    # Try filename in data/csv and relative path from project root.
    for candidate_raw in possible_raw_paths:
        candidate = csv_dir / candidate_raw.name
        if candidate.exists():
            return candidate.resolve()

        candidate = project_root / candidate_raw
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not find CSV file '{csv_arg}'. Tried current folder, project root, and {csv_dir}"
    )


def resolve_lc_path(lc_arg: Optional[str], csv_path: Path, project_root: Path) -> Path:
    """
    Resolve output LC path.
    - If omitted: same folder/name as CSV, with .lc suffix
    - If filename only: write beside CSV
    - If path given: use that path, relative to current folder if needed
    """
    if not lc_arg:
        return csv_path.with_suffix(".lc")

    raw = Path(lc_arg)

    # Filename only -> place beside CSV.
    if raw.parent == Path("."):
        return (csv_path.parent / raw.name).resolve()

    # Relative or absolute path.
    return raw.resolve()


# ----------------------------
# Argument parsing
# ----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CSV path points to an ASTORINO .lc TRANS point file. Paths are auto-resolved."
    )

    parser.add_argument(
        "csv_file",
        nargs="?",
        help="Input CSV file. Optional. If omitted, a numbered CSV selection menu is shown.",
    )
    parser.add_argument(
        "lc_file",
        nargs="?",
        help="Output .lc file. Optional. If omitted, writes beside the selected CSV with the same base name.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all .csv files in data/csv. Ignores positional csv_file/lc_file.",
    )

    parser.add_argument("--prefix", default="P", help="Point name prefix, default: P")
    parser.add_argument("--start-index", type=int, default=1, help="First point index, default: 1")
    parser.add_argument(
        "--name-style",
        choices=["plain", "array", "numbered"],
        default="plain",
        help="plain -> P1, array -> P[1], numbered -> P001. Default: plain",
    )

    # Simple transform parameters.
    parser.add_argument(
        "--base",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help="Robot BASE coordinates of the CSV origin. Default: 0 0 0",
    )
    parser.add_argument(
        "--rot-deg",
        type=float,
        default=0.0,
        help="Rotation around robot/global Z axis in degrees for simple mode. Default: 0",
    )
    parser.add_argument(
        "--align-first-to",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Robot XYZ position for the first CSV point in simple mode. Default: 0 0 0.",
    )

    # 3-point calibration parameters.
    parser.add_argument("--cad-origin", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--cad-x-point", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--cad-y-point", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--robot-origin", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--robot-x-point", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--robot-y-point", nargs=3, type=float, metavar=("X", "Y", "Z"))

    # Z handling.
    parser.add_argument(
        "--z-scale",
        type=float,
        default=1.0,
        help="Scale applied to CSV z before adding to robot position. Default: 1.0",
    )
    parser.add_argument(
        "--z-offset",
        type=float,
        default=0.0,
        help="Extra robot Z/local-normal offset in mm. Default: 0.0",
    )
    parser.add_argument(
        "--fixed-z",
        type=float,
        help="Force all output Z values to this robot BASE Z value after transform.",
    )

    # Orientation handling.
    orient = parser.add_mutually_exclusive_group()
    orient.add_argument(
        "--fixed-oat",
        nargs=3,
        type=float,
        metavar=("O", "A", "T"),
        help="Use fixed ASTORINO O,A,T orientation for every point. Default: 0 0 0.",
    )
    orient.add_argument(
        "--use-csv-oat",
        action="store_true",
        help="Use o,a,t values from the CSV file.",
    )
    parser.add_argument(
        "--add-rot-to-t",
        action="store_true",
        help="When using CSV OAT in simple mode, add --rot-deg to T. Use only if this matches your tool convention.",
    )

    parser.add_argument(
        "--external-axis",
        type=float,
        default=0.0,
        help="Value written after O,A,T in .lc file. Usually 0.000 for ASTORINO. Default: 0.0",
    )
    parser.add_argument(
        "--flag",
        default="1.0",
        help="Final flag value before semicolon in .lc file. Existing ASTORINO exports use 1.0. Default: 1.0",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Decimal places for output coordinates and angles. Default: 3",
    )

    return parser.parse_args()


# ----------------------------
# CSV / LC I/O
# ----------------------------
def get_value_case_insensitive(row: dict, wanted: str) -> str:
    for key, value in row.items():
        if key is not None and key.strip().lower() == wanted.lower():
            return value
    raise KeyError(wanted)


def read_csv_points(path: Path) -> List[CsvPoint]:
    required = {"x", "y", "z", "o", "a", "t"}
    points: List[CsvPoint] = []

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header row.")

        fields = {name.strip().lower() for name in reader.fieldnames if name is not None}
        missing = required - fields
        if missing:
            raise ValueError(f"CSV file is missing required columns: {', '.join(sorted(missing))}")

        for line_no, row in enumerate(reader, start=2):
            try:
                points.append(
                    CsvPoint(
                        x=float(get_value_case_insensitive(row, "x")),
                        y=float(get_value_case_insensitive(row, "y")),
                        z=float(get_value_case_insensitive(row, "z")),
                        o=float(get_value_case_insensitive(row, "o")),
                        a=float(get_value_case_insensitive(row, "a")),
                        t=float(get_value_case_insensitive(row, "t")),
                    )
                )
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"Invalid numeric value on CSV line {line_no}: {row}") from exc

    if not points:
        raise ValueError("CSV contains no points.")
    return points


# ----------------------------
# Vector helpers
# ----------------------------
def vec_sub(a: Iterable[float], b: Iterable[float]) -> Tuple[float, ...]:
    return tuple(x - y for x, y in zip(a, b))


def vec_add(a: Iterable[float], b: Iterable[float]) -> Tuple[float, ...]:
    return tuple(x + y for x, y in zip(a, b))


def vec_mul(s: float, v: Iterable[float]) -> Tuple[float, ...]:
    return tuple(s * x for x in v)


def vec_cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vec_norm(v: Iterable[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def vec_normalize(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    n = vec_norm(v)
    if n < 1e-9:
        raise ValueError("Cannot normalize a zero-length vector.")
    return (v[0] / n, v[1] / n, v[2] / n)


def solve_2x2(a11: float, a12: float, a21: float, a22: float, b1: float, b2: float) -> Tuple[float, float]:
    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-9:
        raise ValueError("CAD calibration points are collinear or too close together.")
    u = (b1 * a22 - a12 * b2) / det
    v = (a11 * b2 - b1 * a21) / det
    return u, v


# ----------------------------
# Coordinate transforms
# ----------------------------
def has_full_calibration(args: argparse.Namespace) -> bool:
    fields = [
        args.cad_origin,
        args.cad_x_point,
        args.cad_y_point,
        args.robot_origin,
        args.robot_x_point,
        args.robot_y_point,
    ]
    return all(value is not None for value in fields)


def has_partial_calibration(args: argparse.Namespace) -> bool:
    fields = [
        args.cad_origin,
        args.cad_x_point,
        args.cad_y_point,
        args.robot_origin,
        args.robot_x_point,
        args.robot_y_point,
    ]
    return any(value is not None for value in fields) and not all(value is not None for value in fields)


def simple_transform(points: List[CsvPoint], args: argparse.Namespace) -> List[Tuple[float, float, float]]:
    theta = math.radians(args.rot_deg)
    c = math.cos(theta)
    s = math.sin(theta)

    base_x, base_y, base_z = args.base

    # If requested, shift base so the first CSV point maps to a chosen robot position.
    if args.align_first_to is not None:
        first = points[0]
        rx0 = c * first.x - s * first.y
        ry0 = s * first.x + c * first.y
        rz0 = first.z * args.z_scale
        base_x = args.align_first_to[0] - rx0
        base_y = args.align_first_to[1] - ry0
        base_z = args.align_first_to[2] - rz0 - args.z_offset

    out = []
    for p in points:
        x = base_x + c * p.x - s * p.y
        y = base_y + s * p.x + c * p.y
        z = base_z + p.z * args.z_scale + args.z_offset
        if args.fixed_z is not None:
            z = args.fixed_z
        out.append((x, y, z))
    return out


def calibrated_transform(points: List[CsvPoint], args: argparse.Namespace) -> List[Tuple[float, float, float]]:
    cad_o = tuple(args.cad_origin)
    cad_x = tuple(args.cad_x_point)
    cad_y = tuple(args.cad_y_point)
    rob_o = tuple(args.robot_origin)
    rob_x = tuple(args.robot_x_point)
    rob_y = tuple(args.robot_y_point)

    cx = (cad_x[0] - cad_o[0], cad_x[1] - cad_o[1])
    cy = (cad_y[0] - cad_o[0], cad_y[1] - cad_o[1])
    rx = vec_sub(rob_x, rob_o)
    ry = vec_sub(rob_y, rob_o)

    normal = vec_normalize(vec_cross(rx, ry))

    out = []
    for p in points:
        dx = p.x - cad_o[0]
        dy = p.y - cad_o[1]
        u, v = solve_2x2(cx[0], cy[0], cx[1], cy[1], dx, dy)

        xyz = rob_o
        xyz = vec_add(xyz, vec_mul(u, rx))
        xyz = vec_add(xyz, vec_mul(v, ry))
        xyz = vec_add(xyz, vec_mul(p.z * args.z_scale + args.z_offset, normal))

        x, y, z = xyz
        if args.fixed_z is not None:
            z = args.fixed_z
        out.append((x, y, z))
    return out


# ----------------------------
# Output point generation
# ----------------------------
def make_name(prefix: str, index: int, style: str) -> str:
    if style == "array":
        return f"{prefix}[{index}]"
    if style == "numbered":
        return f"{prefix}{index:03d}"
    return f"{prefix}{index}"


def make_robot_points(
    points: List[CsvPoint],
    xyz_values: List[Tuple[float, float, float]],
    args: argparse.Namespace,
) -> List[RobotPoint]:
    robot_points: List[RobotPoint] = []
    for offset, (p, xyz) in enumerate(zip(points, xyz_values)):
        index = args.start_index + offset
        name = make_name(args.prefix, index, args.name_style)

        if args.fixed_oat is not None:
            o, a, t = args.fixed_oat
        else:
            o, a, t = p.o, p.a, p.t
            if args.add_rot_to_t:
                t += args.rot_deg

        robot_points.append(RobotPoint(name=name, x=xyz[0], y=xyz[1], z=xyz[2], o=o, a=a, t=t))
    return robot_points


def write_lc(path: Path, robot_points: List[RobotPoint], args: argparse.Namespace) -> None:
    fmt = f"{{:.{args.decimals}f}}"
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(".JOINTS\r\n")
        f.write(".END\r\n")
        f.write(".TRANS\r\n")
        for p in robot_points:
            values = [
                fmt.format(p.x),
                fmt.format(p.y),
                fmt.format(p.z),
                fmt.format(p.o),
                fmt.format(p.a),
                fmt.format(p.t),
                fmt.format(args.external_axis),
                str(args.flag),
            ]
            f.write(p.name + "\t" + "\t".join(values) + "\t;\r\n")
        f.write(".END\r\n")


def convert_one(csv_path: Path, lc_path: Path, args: argparse.Namespace) -> int:
    points = read_csv_points(csv_path)
    if has_full_calibration(args):
        xyz_values = calibrated_transform(points, args)
    else:
        xyz_values = simple_transform(points, args)

    robot_points = make_robot_points(points, xyz_values, args)
    write_lc(lc_path, robot_points, args)

    print(f"Wrote {len(robot_points)} TRANS points")
    print(f"  CSV: {csv_path}")
    print(f"  LC:  {lc_path}")
    print(f"  First point: {robot_points[0]}")
    print(f"  Last point:  {robot_points[-1]}")
    print(f"  Align first to: {args.align_first_to}")
    if args.fixed_oat is not None:
        print(f"  Fixed OAT: {args.fixed_oat}")
    else:
        print("  OAT source: CSV")
    return len(robot_points)


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    args = parse_args()
    project_root = find_project_root()
    csv_dir = default_csv_dir(project_root)

    if has_partial_calibration(args):
        print(
            "ERROR: 3-point calibration mode requires all of these arguments:\n"
            "  --cad-origin --cad-x-point --cad-y-point --robot-origin --robot-x-point --robot-y-point",
            file=sys.stderr,
        )
        return 2

    # New default behavior:
    # - first CSV point becomes robot XYZ 0,0,0 unless --align-first-to is given
    # - all output points use fixed O,A,T 0,0,0 unless --fixed-oat or --use-csv-oat is given
    if args.align_first_to is None:
        args.align_first_to = (0.0, 0.0, 0.0)

    if args.fixed_oat is None and not args.use_csv_oat:
        args.fixed_oat = (0.0, 0.0, 0.0)

    try:
        if args.all:
            csv_files = sorted(csv_dir.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No .csv files found in {csv_dir}")

            print(f"Project root: {project_root}")
            print(f"Converting all CSV files in: {csv_dir}\n")

            total = 0
            for csv_path in csv_files:
                lc_path = csv_path.with_suffix(".lc")
                total += convert_one(csv_path.resolve(), lc_path.resolve(), args)
                print("")
            print(f"Done. Converted {len(csv_files)} file(s), {total} point(s) total.")
            return 0

        csv_path = resolve_csv_path(args.csv_file, project_root)
        lc_path = resolve_lc_path(args.lc_file, csv_path, project_root)

        print(f"Project root: {project_root}")
        convert_one(csv_path, lc_path, args)
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
