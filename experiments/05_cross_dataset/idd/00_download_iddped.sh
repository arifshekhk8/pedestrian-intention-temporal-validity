#!/usr/bin/env bash
# 00_download_iddped.sh — fetch the IDD-PeD annotations from the OFFICIAL CVIT source.
#
# Source of record (linked from the ICRA-2025 paper's project page,
# https://cvit.iiit.ac.in/research/projects/cvit-projects/iddped, and from the official
# code repo https://github.com/Ruthvik9/IDD-PeD):
#   https://cvit.iiit.ac.in/images/datasets/IDDPed/Annotations/annotations.tar          (478,209,024 B)
#   https://cvit.iiit.ac.in/images/datasets/IDDPed/Annotations/annotations_vehicle.tar   (58,593,280 B)
# License: CC BY 4.0 (stated in the paper). No registration/access form required.
#
# We deliberately download ONLY the annotation tars. The video tars (gp_set_0001..0009,
# ~100 h of footage) are NOT needed: the main experiment consumes bounding boxes +
# ego-vehicle OBD speed, both of which live in the annotations. See
# reports/IDD_PeD_schema_audit.md.
#
# The CVIT host throttles a single connection (~350 KB/s measured 2026-08-25) but DOES
# honour HTTP range requests (verified: 206 + content-range). We therefore use the same
# parallel segmented-download strategy this project already uses for the PIE clips
# (pipeline/PROGRESS_LOG.md, Phase 4).
#
# Usage (from the repo root):
#   bash idd_ped_crossdataset/scripts/00_download_iddped.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW="$(cd "$HERE/.." && pwd)/data/raw"
BASE="https://cvit.iiit.ac.in/images/datasets/IDDPed/Annotations"
NPARTS=16

mkdir -p "$RAW"

fetch_parallel() {
    local name="$1" url="$BASE/$1" out="$RAW/$1"
    local total part_size i start end pids=()

    total=$(curl -sIL --max-time 60 "$url" | awk 'BEGIN{IGNORECASE=1}/^content-length:/{v=$2} END{gsub(/\r/,"",v); print v}')
    if [[ -z "$total" || "$total" -le 0 ]]; then
        echo "!! could not determine size of $name" >&2; return 1
    fi
    echo ">> $name : $total bytes, ${NPARTS}-way parallel range download"

    part_size=$(( (total + NPARTS - 1) / NPARTS ))
    rm -f "$out.part."*
    for ((i = 0; i < NPARTS; i++)); do
        start=$(( i * part_size ))
        end=$(( start + part_size - 1 ))
        (( end >= total )) && end=$(( total - 1 ))
        (( start > end )) && continue
        curl -sSL --retry 5 --retry-delay 2 -r "${start}-${end}" -o "$out.part.$(printf '%02d' "$i")" "$url" &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done

    cat "$out.part."* > "$out"
    rm -f "$out.part."*

    local got
    got=$(wc -c < "$out" | tr -d ' ')
    if [[ "$got" != "$total" ]]; then
        echo "!! SIZE MISMATCH for $name: expected $total, got $got" >&2; return 1
    fi
    echo ">> $name : OK ($got bytes)"
}

fetch_parallel annotations_vehicle.tar
fetch_parallel annotations.tar

echo
echo "=== checksums (recorded in manifests/dataset_manifest.json) ==="
shasum -a 256 "$RAW/annotations.tar" "$RAW/annotations_vehicle.tar"
