"""GPS point logs (CSV/TXT with latitude, longitude columns) -> KML polygons. Needs the [utils] extra.

    python -m vm0047_pb.utils.gps_to_kml INPUT OUTPUT      # file->file, folder->folder, zip->zip
"""
from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

LAT_COL, LON_COL = "latitude", "longitude"


def read_points(source) -> list[tuple[float, float]]:
    """Return [(lon, lat), ...] from a path, text file object or bytes stream.

    Rows with empty or non-numeric coordinates are skipped; out-of-range values raise.
    """
    if isinstance(source, (str, Path)):
        text = Path(source).read_text(encoding="utf-8-sig")
    elif isinstance(source, bytes):
        text = source.decode("utf-8-sig")
    else:
        raw = source.read()
        text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or LAT_COL not in reader.fieldnames or LON_COL not in reader.fieldnames:
        raise ValueError(f"input needs '{LAT_COL}' and '{LON_COL}' columns, found {reader.fieldnames}")
    pts = []
    for row in reader:
        try:
            lat, lon = float(row[LAT_COL]), float(row[LON_COL])
        except (TypeError, ValueError):
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"coordinate out of range: lat={lat} lon={lon}")
        pts.append((lon, lat))
    return pts


def dedupe_consecutive(points):
    """Drop consecutive repeats (stationary GPS fixes)."""
    out = []
    for p in points:
        if not out or p != out[-1]:
            out.append(p)
    return out


def build_kml(points, name: str = "Land Plot Polygon"):
    """simplekml.Kml with one closed, styled polygon (red outline, translucent green fill)."""
    import simplekml

    if len(points) < 3:
        raise ValueError("need at least 3 points to form a polygon")
    ring = list(points) + ([points[0]] if points[0] != points[-1] else [])
    kml = simplekml.Kml()
    pol = kml.newpolygon(name=name, outerboundaryis=ring)
    pol.style.linestyle.color = simplekml.Color.red
    pol.style.linestyle.width = 2
    pol.style.polystyle.color = simplekml.Color.changealphaint(100, simplekml.Color.green)
    return kml


def txt_to_kml(txt_path, kml_path, dedupe: bool = True):
    pts = read_points(txt_path)
    if dedupe:
        pts = dedupe_consecutive(pts)
    build_kml(pts, name=f"Land Plot Polygon - {Path(txt_path).stem}").save(str(kml_path))
    return kml_path


def batch_to_kml(input_path, output_path, dedupe: bool = True):
    """Folder of .txt/.csv -> folder of .kml, or a zip of them -> a zip of .kml.

    Returns (n_ok, {name: error}). One bad file does not stop the batch.
    """
    ok, errors = 0, {}

    def convert(name, stream):
        pts = read_points(stream)
        if dedupe:
            pts = dedupe_consecutive(pts)
        return build_kml(pts, name=f"Land Plot Polygon - {Path(name).stem}").kml()

    in_p, out_p = Path(input_path), Path(output_path)
    if in_p.suffix.lower() == ".zip":
        with zipfile.ZipFile(in_p) as zin, zipfile.ZipFile(out_p, "w", zipfile.ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                if info.filename.lower().endswith((".txt", ".csv")):
                    try:
                        zout.writestr(Path(info.filename).with_suffix(".kml").as_posix(), convert(info.filename, zin.read(info)))
                        ok += 1
                    except Exception as exc:
                        errors[info.filename] = str(exc)
    else:
        out_p.mkdir(parents=True, exist_ok=True)
        for f in sorted(in_p.iterdir()):
            if f.suffix.lower() in (".txt", ".csv"):
                try:
                    (out_p / f"{f.stem}.kml").write_text(convert(f.name, f.read_bytes()), encoding="utf-8")
                    ok += 1
                except Exception as exc:
                    errors[f.name] = str(exc)
    return ok, errors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--keep-duplicates", action="store_true", help="keep consecutive identical fixes")
    a = ap.parse_args(argv)
    p = Path(a.input)
    if p.is_file() and p.suffix.lower() != ".zip":
        txt_to_kml(p, a.output, dedupe=not a.keep_duplicates)
        print("wrote", a.output)
        return 0
    ok, errors = batch_to_kml(p, a.output, dedupe=not a.keep_duplicates)
    print(f"{ok} converted, {len(errors)} failed")
    for k, v in errors.items():
        print("FAILED:", k, "-", v)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
