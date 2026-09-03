"""01_build_database.py — parse the IDD-PeD CVAT annotations into one database pickle.

Reads   data/iddped/annotations/**            (33 video XMLs)
        data/iddped/annotations_vehicle/**    (34 OBD XMLs)
Writes  data/iddped_database.pkl              (this folder only)
        logs/01_build_database.log            (via shell redirection)

Run from the repo root:
    python idd_ped_crossdataset/scripts/01_build_database.py
"""
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FOLDER = HERE.parent
sys.path.insert(0, str(HERE / "lib"))

from iddped_parser import build_database  # noqa: E402

ANN = FOLDER / "data" / "iddped" / "annotations"
VEH = FOLDER / "data" / "iddped" / "annotations_vehicle"
OUT = FOLDER / "data" / "iddped_database.pkl"


def main():
    print("Parsing IDD-PeD annotations …")
    db, issues = build_database(ANN, VEH, verbose=True)

    n_vid = sum(len(v) for v in db.values())
    n_ped = sum(len(vv["pedestrian_annotations"]) for v in db.values() for vv in v.values())
    print(f"\nsets {len(db)} | videos {n_vid} | pedestrian tracks {n_ped}")
    print(f"issues: missing OBD file for {len(issues['missing_obd_file'])} video(s) "
          f"{issues['missing_obd_file']}")
    print(f"        pedestrians without POI attributes: {len(issues['ped_without_attributes'])}")

    with open(OUT, "wb") as f:
        pickle.dump({"database": db, "issues": issues}, f, pickle.HIGHEST_PROTOCOL)
    print(f"\nWrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
