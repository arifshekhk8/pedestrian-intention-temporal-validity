"""iddped_parser.py — self-contained parser for the IDD-PeD CVAT annotations.

ISOLATED COPY NOTICE
--------------------
This module re-implements, in a minimal and dependency-free form, the three parsing
functions of the dataset authors' official interface
(`Intention/iddped_interface.py` in https://github.com/Ruthvik9/IDD-PeD, ICRA 2025):

    _get_annotations(setid, vid)        -> per-frame boxes, occlusion, behavior tags
    _get_ped_attributes(setid, vid)     -> per-pedestrian track attributes (incl. crossing_point)
    _get_vehicle_attributes(setid, vid) -> per-frame OBD record (incl. OBD_speed)

We re-implement rather than import because the official file is a 2,258-line class that
pulls in OpenCV/MMPose/TensorFlow-era helpers, hardcodes its own data paths, and carries
PIE-inherited dead code. The label->scalar maps and the element/attribute paths below are
copied VERBATIM from that file so the parse is bit-identical in the fields we use; a
verification pass in `scripts/01_build_database.py` cross-checks our output against the
authors' released `iddp_database.pkl` where available.

Nothing in this file writes outside `idd_ped_crossdataset/`.

Key schema facts (verified against the released data, see reports/IDD_PeD_schema_audit.md):
  * Per-pedestrian attributes live on tracks with label "POI", NOT on the "pedestrian"
    track — this is an IDD-PeD quirk. `crossing_point` and `crossing` are POI attributes.
  * Boxes with outside="1" are dropped (pedestrian has left the frame).
  * Image size is per-video (`meta/task/original_size`) and is NOT constant across the
    dataset (1920x1080 and 1920x1440 both occur).
  * The `ddpai` (25 fps) camera directories exist but are EMPTY in the public release;
    everything published is `gopro` at 30 fps.
"""
from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree as ET

# ---------------------------------------------------------------- label maps
# Copied verbatim from iddped_interface.py::_map_text_to_scalar
TEXT_TO_SCALAR = {
    "occlusion": {"None": 0, "Part": 1, "Full": 2},
    "CrossingBehavior": {"CU": 0, "CFU": 1, "CD": 2, "CFD": 3, "N/A": 4, "CI": -1},
    "TrafficInteraction": {"WTT": 0, "HG": 1, "Other": 2, "N/A": 3},
    "PedestrianActivity": {"Walking": 0, "MS": 1, "N/A": 2},
    "AttentionIndicators": {"LOS": 0, "FTT": 1, "NL": 2, "DB": 3},
    "SocialDynamics": {"GS": 0, "CFA": 1, "AWC": 2, "N/A": 3},
    "StationaryBehavior": {"Sitting": 0, "Standing": 1, "IWA": 2, "Other": 3, "N/A": 4},
    "crossing": {"no": 0, "yes": 1},
    "age": {"child": 0, "teenager": 1, "adult": 2, "senior": 3},
    "carrying_object": {"none": 0, "small": 1, "large": 2},
    "crossing_motive": {"yes": 0, "maybe": 1, "no": 2},
    "crosswalk_usage": {"yes": 0, "no": 1, "partial": 2, "N/A": 3},
    "intersection_type": {"NI": 0, "U-turn": 1, "T-right": 2, "T-left": 3,
                          "four-way": 4, "Y-intersection": 5},
    "motion_direction": {"OW": 0, "TW": 1},
    "signalized_type": {"N/A": 0, "C": 1, "S": 2, "CS": 3},
    "road_type": {"main": 0, "secondary": 1, "street": 2, "lane": 3},
    "location_type": {"urban": 0, "rural": 1, "commercial": 2, "residential": 3},
    "vehicle": {"car": 0, "motorcycle": 1, "bicycle": 2, "auto": 3, "bus": 4,
                "cart": 5, "truck": 6, "other": 7},
    "traffic_light": {"pedestrian": 0, "vehicle": 1},
    "state": {"red": 0, "orange": 1, "green": 2},
}

# CVAT per-frame attribute short names -> the behavior category they encode
BEH_MAPPER = {"CrossingBehavior": "CB", "TrafficInteraction": "TI",
              "PedestrianActivity": "PA", "AttentionIndicators": "AI",
              "SocialDynamics": "SD", "StationaryBehavior": "SB"}

# CrossingBehavior scalars that mean "this pedestrian is crossing the road right now".
# CU/CFU = crossing undesignated (jaywalking), CD/CFD = crossing at a designated place.
CROSSING_BEHAVIOR_SCALARS = {0, 1, 2, 3}
# CI = "crossing the road but NOT in the path of the ego-vehicle" — treated separately;
# see reports/temporal_protocol_IDD_PeD.md for the decision and its justification.
CROSSING_IRRELEVANT_SCALAR = -1
NOT_CROSSING_SCALAR = 4  # "N/A"


def _map(label_type: str, value):
    """Text -> scalar, tolerating unseen/None values by returning None (recorded, never guessed)."""
    if value is None:
        return None
    return TEXT_TO_SCALAR.get(label_type, {}).get(value)


def cam_of(set_id: str) -> str:
    """'gp_set_0003' -> 'gopro'  (the authors' own mapping)."""
    prefix = set_id.split("_")[0]
    return {"gp": "gopro", "d": "ddpai"}.get(prefix, prefix)


def list_videos(annotation_root: Path):
    """Yield (set_id, video_id, xml_path) for every released annotation XML, sorted."""
    out = []
    for xml in sorted(annotation_root.rglob("*.xml")):
        m = re.match(r"(gp_set_\d+|d_set_\d+)_vid_\d+$", xml.stem)
        if not m:
            continue
        out.append((m.group(1), xml.stem, xml))
    return out


# ---------------------------------------------------------------- per-video parse
def parse_video_annotations(xml_path: Path) -> dict:
    """Mirror of iddped_interface._get_annotations for the fields this study uses."""
    root = ET.parse(xml_path).getroot()
    task = root.find("./meta/task")
    orig = task.find("original_size")

    ann = {
        "num_frames": int(task.find("size").text),
        "width": int(orig.find("width").text) if orig is not None else None,
        "height": int(orig.find("height").text) if orig is not None else None,
        "pedestrian_annotations": {},
    }

    for t in root.findall("./track"):
        if t.get("label") != "pedestrian":
            continue
        boxes = t.findall("./box")
        if not boxes:
            continue
        id_el = boxes[0].find('./attribute[@name="id"]')
        obj_id = id_el.text if id_el is not None else None
        if obj_id is None:
            continue  # the authors skip these too (annotator error)

        rec = {"frames": [], "bbox": [], "occlusion": [],
               "behavior": {k: [] for k in BEH_MAPPER}}
        for b in boxes:
            if int(b.get("outside")) == 1:
                continue  # pedestrian outside the frame
            rec["bbox"].append([float(b.get("xtl")), float(b.get("ytl")),
                                float(b.get("xbr")), float(b.get("ybr"))])
            occ_el = b.find('./attribute[@name="occlusion"]')
            rec["occlusion"].append(_map("occlusion", occ_el.text if occ_el is not None else None))
            rec["frames"].append(int(b.get("frame")))
            for beh, short in BEH_MAPPER.items():
                el = b.find(f'./attribute[@name="{short}"]')
                rec["behavior"][beh].append(_map(beh, el.text if el is not None else None))
        if rec["frames"]:
            # A CVAT track can list boxes out of order; the downstream builder sorts,
            # but keep the parse faithful and record whether it was already sorted.
            ann["pedestrian_annotations"][obj_id] = rec

    return ann


def parse_ped_attributes(xml_path: Path) -> dict:
    """Mirror of iddped_interface._get_ped_attributes.

    IDD-PeD stores per-pedestrian track attributes on a *separate* track whose label is
    'POI', keyed by the same pedestrian id. Integer-valued attributes (group_size,
    crossing_point) are int()-cast; the rest go through the label map.
    """
    root = ET.parse(xml_path).getroot()
    attributes = {}
    for t in root.findall("./track"):
        if t.get("label") != "POI":
            continue
        boxes = t.findall("./box")
        if not boxes:
            continue
        id_el = boxes[0].find('./attribute[@name="id"]')
        if id_el is None or id_el.text is None:
            continue
        ped_id = id_el.text
        rec = {}
        for attribute in boxes[0].findall("./attribute"):
            k = attribute.get("name")
            v = attribute.text
            if "id" in k:
                continue
            try:
                rec[k] = int(v)
            except (TypeError, ValueError):
                rec[k] = _map(k, v)
        attributes[ped_id] = rec
    return attributes


def parse_vehicle_attributes(xml_path: Path) -> dict:
    """Mirror of iddped_interface._get_vehicle_attributes: {frame_id: {OBD_speed, accT, accX, accY, accZ}}."""
    root = ET.parse(xml_path).getroot()
    out = {}
    for f in root.findall("./frame"):
        out[int(f.get("id"))] = {k: float(v) for k, v in f.attrib.items() if k != "id"}
    return out


def build_database(annotation_root: Path, vehicle_root: Path, verbose: bool = True) -> dict:
    """Assemble the full {set_id: {video_id: {...}}} database, PIE-interface style.

    Missing vehicle files and missing POI attributes are RECORDED, never silently
    imputed — the counts land in reports/IDD_PeD_schema_audit.md.
    """
    db, issues = {}, {"missing_obd_file": [], "ped_without_attributes": [],
                      "obd_missing_frames": []}

    for set_id, video_id, xml in list_videos(annotation_root):
        ann = parse_video_annotations(xml)
        ann["attributes_by_ped"] = parse_ped_attributes(xml)

        obd_path = vehicle_root / cam_of(set_id) / set_id / f"{video_id}_obd.xml"
        if obd_path.exists():
            ann["vehicle_annotations"] = parse_vehicle_attributes(obd_path)
        else:
            ann["vehicle_annotations"] = {}
            issues["missing_obd_file"].append(video_id)

        for ped, rec in ann["pedestrian_annotations"].items():
            attrs = ann["attributes_by_ped"].get(ped)
            if attrs is None:
                issues["ped_without_attributes"].append((video_id, ped))
            rec["attributes"] = attrs

        db.setdefault(set_id, {})[video_id] = ann
        if verbose:
            n_ped = len(ann["pedestrian_annotations"])
            print(f"  {video_id}: {ann['num_frames']:>6} frames  "
                  f"{ann['width']}x{ann['height']}  {n_ped:>4} peds  "
                  f"{len(ann['vehicle_annotations']):>6} obd rows")

    return db, issues
