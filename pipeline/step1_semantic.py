#!/usr/bin/env python3
"""
Step 1: Convert a raw GLB scene to HM3D-compatible semantic GLB.

Adds per-vertex COLOR_0 attribute (VEC3, UNSIGNED_BYTE, normalized)
encoding the semantic object color, and generates semantic.txt.

Usage:
    python3 step1_semantic.py
"""

import json
import struct
import sys
from pathlib import Path

# Import config from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    SCENE_GLB, OUTPUT_DIR, SCENE_NAME, SCENE_CONFIG_FILE,
    SEMANTIC_ANNOTATIONS, SCENE_DATASET_CONFIG,
    derived_paths,
)


# ─── Helpers ───────────────────────────────────────────────────────────────

def object_id_to_hex(obj_id):
    """Deterministic 6-char hex color from object_id."""
    r = (obj_id * 67 + 13) % 256
    g = (obj_id * 113 + 47) % 256
    b = (obj_id * 173 + 89) % 256
    return f"{r:02X}{g:02X}{b:02X}"


def hex_to_rgb(hex_color):
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def make_hm3d_mesh_name(mesh_idx):
    obj_id, label, region = SEMANTIC_ANNOTATIONS[mesh_idx]
    return f"chunk{mesh_idx:03d}_group000_sub{region:03d}_object{label}{obj_id:03d}_1"


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    paths = derived_paths()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Step 1: Semantic GLB Conversion")
    print(f"  Input:  {SCENE_GLB}")
    print(f"  Output: {paths['semantic_glb']}")
    print(f"  TXT:    {paths['semantic_txt']}")
    print()

    # ── Parse GLB ─────────────────────────────────────────────────────────
    with open(SCENE_GLB, 'rb') as f:
        magic = f.read(4)
        assert magic == b'glTF', f"Not a valid GLB file: {SCENE_GLB}"
        version = struct.unpack('<I', f.read(4))[0]
        total_length = struct.unpack('<I', f.read(4))[0]
        print(f"  GLB v{version}, {total_length:,} bytes")

        json_chunk_len = struct.unpack('<I', f.read(4))[0]
        assert f.read(4) == b'JSON'
        json_data = json.loads(f.read(json_chunk_len))
        print(f"  JSON chunk: {json_chunk_len:,} bytes")

        bin_chunk_len = struct.unpack('<I', f.read(4))[0]
        assert f.read(4) == b'BIN\x00'
        bin_data = f.read(bin_chunk_len)
        print(f"  BIN chunk:  {bin_chunk_len:,} bytes")

    meshes = json_data['meshes']
    accessors = json_data['accessors']
    buffer_views = json_data['bufferViews']
    print(f"  Meshes: {len(meshes)}")
    print()

    # ── Validate SEMANTIC_ANNOTATIONS covers all meshes ───────────────────
    if len(SEMANTIC_ANNOTATIONS) != len(meshes):
        print(f"  ⚠  SEMANTIC_ANNOTATIONS has {len(SEMANTIC_ANNOTATIONS)} entries "
              f"but GLB has {len(meshes)} meshes.")
        print(f"  Please update config.py → SEMANTIC_ANNOTATIONS")
        sys.exit(1)

    # ── Add COLOR_0 to each primitive ─────────────────────────────────────
    new_color_data = b''
    new_accessors = []
    new_buffer_views = []

    for mi, mesh in enumerate(meshes):
        obj_id, label, region = SEMANTIC_ANNOTATIONS[mi]
        hex_color = object_id_to_hex(obj_id)
        r8, g8, b8 = hex_to_rgb(hex_color)

        for pi, prim in enumerate(mesh['primitives']):
            pos_acc_idx = prim['attributes']['POSITION']
            vertex_count = accessors[pos_acc_idx]['count']

            # VEC3, UNSIGNED_BYTE — 3 bytes per vertex
            raw = struct.pack('<3B', r8, g8, b8) * vertex_count

            offset_in_new = len(new_color_data)
            new_color_data += raw

            bv_idx = len(buffer_views) + len(new_buffer_views)
            new_buffer_views.append({
                "buffer": 0,
                "byteOffset": len(bin_data) + offset_in_new,
                "byteLength": len(raw),
            })

            acc_idx = len(accessors) + len(new_accessors)
            new_accessors.append({
                "bufferView": bv_idx,
                "componentType": 5121,   # UNSIGNED_BYTE
                "count": vertex_count,
                "type": "VEC3",
                "normalized": True,
            })

            prim['attributes']['COLOR_0'] = acc_idx

    print(f"  Added {len(new_accessors)} COLOR_0 accessors ({len(new_color_data):,} bytes)")

    # ── Update mesh names ──────────────────────────────────────────────────
    for mi, mesh in enumerate(meshes):
        mesh['name'] = make_hm3d_mesh_name(mi)

    # ── Write semantic GLB ────────────────────────────────────────────────
    json_data['accessors'].extend(new_accessors)
    json_data['bufferViews'].extend(new_buffer_views)

    while len(new_color_data) % 4 != 0:
        new_color_data += b'\x00'
    new_bin = bin_data + new_color_data

    json_bytes = json.dumps(json_data).encode('utf-8')
    while len(json_bytes) % 4 != 0:
        json_bytes += b' '

    total = 12 + 8 + len(json_bytes) + 8 + len(new_bin)

    with open(paths['semantic_glb'], 'wb') as f:
        f.write(b'glTF')
        f.write(struct.pack('<I', 2))
        f.write(struct.pack('<I', total))
        f.write(struct.pack('<I', len(json_bytes)))
        f.write(b'JSON')
        f.write(json_bytes)
        f.write(struct.pack('<I', len(new_bin)))
        f.write(b'BIN\x00')
        f.write(new_bin)

    print(f"  ✓ {paths['semantic_glb']} ({total:,} bytes)")

    # ── Write semantic.txt (HM3D format) ──────────────────────────────────
    with open(paths['semantic_txt'], 'w') as f:
        f.write("HM3D Semantic Annotations\n")
        for mi in sorted(SEMANTIC_ANNOTATIONS):
            obj_id, label, region = SEMANTIC_ANNOTATIONS[mi]
            hex_color = object_id_to_hex(obj_id)
            f.write(f'{obj_id},{hex_color},"{label}",{region}\n')

    print(f"  ✓ {paths['semantic_txt']}")

    # ── Write scene_dataset_config.json ───────────────────────────────────
    with open(paths['scene_config'], 'w') as f:
        json.dump(SCENE_DATASET_CONFIG, f, indent=2)

    print(f"  ✓ {paths['scene_config']}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Scene Semantic Objects")
    print(f"{'='*60}")
    print(f"{'ID':>3}  {'Hex':8}  {'Label':15}  Region")
    print(f"{'-'*45}")
    for mi in sorted(SEMANTIC_ANNOTATIONS):
        obj_id, label, region = SEMANTIC_ANNOTATIONS[mi]
        hex_color = object_id_to_hex(obj_id)
        print(f"{obj_id:3d}  #{hex_color}      {label:15}  {region}")

    print(f"\n✓ Step 1 complete.")
    print(f"  semantic_asset:              {paths['semantic_glb'].name}")
    print(f"  semantic_descriptor_filename: {paths['semantic_txt'].name}")


if __name__ == "__main__":
    main()
