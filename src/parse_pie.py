"""
Parse PIE dataset XML annotations into a single flat DataFrame.

Joins:
  - annotations/set*/video_*_annt.xml        → bounding boxes, action, ped_id
  - annotations_attributes/set*/video_*_attributes.xml → crossing_label per ped
  - annotations_vehicle/set*/video_*_obd.xml → OBD_speed per frame

Output columns: set_id, video_id, ped_id, frame, x1, y1, x2, y2,
                vehicle_speed, action, crossing_label
"""

import argparse
import glob
import os
import re
from xml.etree import ElementTree as ET

import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_annt(path: str) -> list[dict]:
    """Parse *_annt.xml → list of per-box dicts."""
    tree = ET.parse(path)
    root = tree.getroot()
    rows = []
    for track in root.findall("track[@label='pedestrian']"):
        for box in track.findall("box"):
            if box.attrib.get("outside") == "1":
                continue  # pedestrian left the frame
            attrs = {a.attrib["name"]: a.text for a in box.findall("attribute")}
            ped_id = attrs.get("id", "")
            action = attrs.get("action", "")
            rows.append(
                {
                    "ped_id": ped_id,
                    "frame": int(box.attrib["frame"]),
                    "x1": float(box.attrib["xtl"]),
                    "y1": float(box.attrib["ytl"]),
                    "x2": float(box.attrib["xbr"]),
                    "y2": float(box.attrib["ybr"]),
                    "action": action,
                }
            )
    return rows


def parse_attributes(path: str) -> dict[str, int]:
    """Parse *_attributes.xml → {ped_id: crossing_label}."""
    tree = ET.parse(path)
    root = tree.getroot()
    mapping = {}
    for ped in root.findall("pedestrian"):
        ped_id = ped.attrib.get("id", "")
        crossing = ped.attrib.get("crossing", "-1")
        mapping[ped_id] = int(crossing)
    return mapping


def parse_obd(path: str) -> dict[int, float]:
    """Parse *_obd.xml → {frame_id: OBD_speed}."""
    tree = ET.parse(path)
    root = tree.getroot()
    mapping = {}
    for frame in root.findall("frame"):
        frame_id = int(frame.attrib["id"])
        speed = frame.attrib.get("OBD_speed")
        mapping[frame_id] = float(speed) if speed is not None else float("nan")
    return mapping


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_dataframe(pie_root: str) -> pd.DataFrame:
    annt_pattern = os.path.join(pie_root, "annotations", "set*", "*_annt.xml")
    annt_files = sorted(glob.glob(annt_pattern))

    if not annt_files:
        raise FileNotFoundError(f"No annotation files found under {pie_root}/annotations/")

    all_rows = []

    for annt_path in tqdm(annt_files, desc="Parsing videos"):
        # Derive set_id and video_id from path
        m = re.search(r"(set\d+)[/\\](video_\d+)_annt\.xml", annt_path)
        if not m:
            continue
        set_id, video_id = m.group(1), m.group(2)

        # Sibling paths for attributes and OBD
        attr_path = os.path.join(
            pie_root, "annotations_attributes", set_id, f"{video_id}_attributes.xml"
        )
        obd_path = os.path.join(
            pie_root, "annotations_vehicle", set_id, f"{video_id}_obd.xml"
        )

        # Parse all three sources
        boxes = parse_annt(annt_path)
        crossing_map = parse_attributes(attr_path) if os.path.exists(attr_path) else {}
        obd_map = parse_obd(obd_path) if os.path.exists(obd_path) else {}

        for row in boxes:
            row["set_id"] = set_id
            row["video_id"] = video_id
            row["crossing_label"] = crossing_map.get(row["ped_id"], -1)
            row["vehicle_speed"] = obd_map.get(row["frame"], float("nan"))

        all_rows.extend(boxes)

    df = pd.DataFrame(
        all_rows,
        columns=[
            "set_id", "video_id", "ped_id", "frame",
            "x1", "y1", "x2", "y2",
            "vehicle_speed", "action", "crossing_label",
        ],
    )

    # Drop pedestrians with crossing_label == -1 (not annotated for intent)
    df = df[df["crossing_label"] != -1].reset_index(drop=True)

    return df


def print_summary(df: pd.DataFrame) -> None:
    total_rows = len(df)
    unique_peds = df["ped_id"].nunique()
    label_counts = df["crossing_label"].value_counts().sort_index()
    missing_speed = df["vehicle_speed"].isna().sum()

    print("\n=== Dataset Summary ===")
    print(f"Total rows          : {total_rows:,}")
    print(f"Unique pedestrians  : {unique_peds:,}")
    print(f"crossing_label == 0 : {label_counts.get(0, 0):,}  (not-crossing)")
    print(f"crossing_label == 1 : {label_counts.get(1, 0):,}  (crossing)")
    print(f"Missing vehicle_speed: {missing_speed:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse PIE dataset into a flat DataFrame.")
    parser.add_argument("--pie-root", default="PIE", help="Path to PIE dataset root")
    parser.add_argument("--out", default="pie_annotations.pkl", help="Output pickle path")
    args = parser.parse_args()

    df = build_dataframe(args.pie_root)
    df.to_pickle(args.out)
    print(f"\nSaved {len(df):,} rows → {args.out}")
    print_summary(df)


if __name__ == "__main__":
    main()
