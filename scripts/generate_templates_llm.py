#!/usr/bin/env python3
"""
LLM-Assisted Layout Template Generation for BioVLN.

Takes a natural-language description of a laboratory layout and uses an LLM
(GPT-4o or DeepSeek) to generate validated layout templates for the procedural
scene generation pipeline (config.py → scene_generator → Blender → episodes).

Usage:
  # Interactive — describe a lab, get a template
  python generate_templates_llm.py --desc "A BSL-2 microbiology lab with biosafety cabinet on east wall"

  # From a file
  python generate_templates_llm.py --file my_lab_desc.txt

  # Save generated template to config.py
  python generate_templates_llm.py --desc "..." --save

  # Generate N variants from one description
  python generate_templates_llm.py --desc "..." --variants 5 --save
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# ─── Instrument Catalog (mirrors config.py) ────────────────────────────────

INSTRUMENT_CATALOG = {
    "refrigerator":    {"zone": "wall",      "size": "0.89×0.91×1.78m", "note": "door faces south (-Y)"},
    "incubator":       {"zone": "wall",      "size": "1.35×0.85×1.90m", "note": "door on wide side"},
    "cabinet":         {"zone": "wall",      "size": "1.38×0.57×1.77m", "note": "doors on wide side"},
    "centrifuge":      {"zone": "table_top", "size": "0.49×0.42×0.63m", "note": "control panel on side"},
    "mini_centrifuge": {"zone": "table_top", "size": "0.66×0.50×0.43m", "note": "compact, lid on top"},
    "pipette":         {"zone": "table_top", "size": "0.04×0.32×0.28m", "note": "very small, on table"},
    "waste_bin":       {"zone": "floor",     "size": "0.38×0.37×0.51m", "note": "small, floor-standing anywhere"},
}

FURNITURE_CATALOG = {
    "lab_bench":   {"size": "1.55×5.94×0.76m", "note": "long bench, goes against a wall"},
    "lab_table":   {"size": "2.40×0.90×1.10m", "note": "large table, center of room"},
}

ROOM_DIMS = {"width": 8.0, "depth": 6.0, "height": 3.24}
WALLS = ["north", "south", "east", "west"]

# ─── System Prompt ──────────────────────────────────────────────────────────

def build_system_prompt() -> str:
    """Build the system prompt encoding all BioVLN domain knowledge."""

    inst_lines = []
    for name, info in INSTRUMENT_CATALOG.items():
        inst_lines.append(
            f"  - {name}: zone={info['zone']}, size={info['size']}, {info['note']}")

    return f"""You are a laboratory architect designing layouts for a biology lab simulation platform.

## AVAILABLE INSTRUMENTS (you MUST only use these names)
{chr(10).join(inst_lines)}

## AVAILABLE FURNITURE
  - lab_bench: long wall bench ({FURNITURE_CATALOG['lab_bench']['size']})
  - lab_table: large central table ({FURNITURE_CATALOG['lab_table']['size']})

## ROOM CONSTRAINTS
  - Single room: {ROOM_DIMS['width']}×{ROOM_DIMS['depth']}m inner, {ROOM_DIMS['height']}m high
  - Multi-room: rooms connected by internal doorways, automatic offset calculation
  - Walls: north (-Y), south (+Y), east (+X), west (-X)
  - Door to outside: typically on south wall
  - Wall instruments sit flush against the wall face, facing into the room
  - Table-top instruments sit ON furniture surfaces (lab_bench or lab_table)
  - Floor instruments (waste_bin) can go anywhere against a wall or in corners
  - Minimum 0.5m clearance between instruments (Zone 2 constraint)

## LABORATORY DOMAIN KNOWLEDGE
  - Refrigerators: cold storage, typically on north wall (away from windows) or in corners
  - Incubators: temperature-controlled, often paired with or opposite refrigerators
  - Biosafety cabinets: containment work, on east or west walls for workflow separation
  - Centrifuges: on sturdy benches/tables, vibration-sensitive, avoid walls
  - Mini-centrifuges: compact, on tables near analytical workstations
  - Pipettes: small bench-top items, near centrifuges or sample prep areas
  - Waste bins: near exits, under benches, or in corners
  - Workflow: sample in → prep area (centrifuge, pipette) → analysis (incubator) → storage (refrigerator) → waste

## LAYOUT PRINCIPLES
  1. Separate "wet" work (centrifuges, pipettes) from "dry" storage (refrigerators, cabinets)
  2. Place large wall instruments first, then fill gaps with smaller instruments
  3. Table instruments should cluster by workflow (e.g. centrifuge + pipette together)
  4. Waste bins near exits or corners, not in prime wall space
  5. Multi-room layouts: separate noisy/cold equipment from main workspace
  6. Always leave at least one wall segment open for circulation

## OUTPUT FORMAT (valid JSON only, no markdown, no explanation)

You MUST output valid JSON matching this exact schema:

{{
  "template_name": "snake_case_name",
  "description": "one sentence describing this layout's intent",
  "room": {{"width": 8.0, "depth": 6.0, "height": 3.24, "wall_thickness": 0.0}},
  "doorways": [{{"wall": "south", "position": 0.5, "width": 1.2, "height": 2.1}}],
  "furniture": {{
    "<furniture_id>": {{"catalog_key": "lab_bench|lab_table",
                        "wall": "north|south|east|west" (only if bench),
                        "offset": 0.08 (only if bench),
                        "zone": "center" (only if table)}}
  }},
  "instrument_slots": {{
    "<slot_id>": {{
      "allowed": ["instrument_name", ...],
      "count": <N> or [<min>, <max>],
      "zone": "wall|table_top|floor|center",
      "wall": "north|south|east|west" (REQUIRED if zone=wall or zone=floor),
      "position_hint": 0.0-1.0 (only if zone=wall),
      "on_furniture": "<furniture_id>" (REQUIRED if zone=table_top)
    }}
  }}
}}

## SLOT DESIGN RULES
  - Each slot's "allowed" list MUST only contain names from the AVAILABLE INSTRUMENTS list above
  - Each furniture entry MUST use a catalog_key from AVAILABLE FURNITURE above
  - "count": use integer for mandatory (e.g. 1), use [0, N] for optional
  - "zone": "wall" for wall instruments, "table_top" for instruments on furniture, "floor" for waste_bin on floor, "center" for instruments on open floor
  - Wall instruments MUST have "wall" field (north/south/east/west)
  - Table-top instruments MUST have "on_furniture" referencing a furniture_id
  - At least 3 instrument slots, at most 10
  - Total instrument count (sum of max count) should be 4-8 per single room
  - position_hint distributes wall instruments along the wall (0.0=leftmost, 0.5=center, 1.0=rightmost)

Output ONLY the JSON object. No surrounding text, no markdown code fences."""


def build_user_prompt(user_description: str, template_name: Optional[str] = None) -> str:
    """Build the user prompt from natural language description."""
    name_hint = f'\nGenerate a template named "{template_name}".' if template_name else ""
    return f"""Design ONE BioVLN laboratory layout matching this description:

"{user_description}"
{name_hint}

Remember: output ONLY the JSON object, no explanation."""


# ─── LLM Client ─────────────────────────────────────────────────────────────

def call_llm(system_prompt: str, user_prompt: str,
             use_openai: bool = True) -> Optional[dict]:
    """Send prompts to LLM, return parsed JSON dict or None."""

    # Clear proxy env vars
    for k in ["ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "http_proxy", "https_proxy"]:
        os.environ.pop(k, None)

    if use_openai:
        return _call_openai(system_prompt, user_prompt)
    else:
        return _call_deepseek(system_prompt, user_prompt)


def _call_openai(system_prompt: str, user_prompt: str) -> Optional[dict]:
    """Call GPT-4o via jeniya.cn proxy."""
    try:
        import openai
        client = openai.OpenAI(
            api_key="YOUR_API_KEY_HERE",
            base_url="https://jeniya.cn/v1",
            timeout=30,
        )
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2000,
            temperature=0.7,
        )
        text = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ✗ GPT-4o API error: {e}", file=sys.stderr)
        return None

    return _parse_llm_json(text)


def _call_deepseek(system_prompt: str, user_prompt: str) -> Optional[dict]:
    """Call DeepSeek via Anthropic-compatible API."""
    import requests
    try:
        resp = requests.post(
            "https://api.deepseek.com/anthropic/v1/messages",
            json={
                "model": "deepseek-chat",
                "max_tokens": 2000,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            },
            headers={
                "x-api-key": "sk-8420c8a3c06f4cb7a9508c20bf79442d",
                "anthropic-version": "2023-06-01",
            },
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"  ✗ DeepSeek API error: {resp.status_code} {resp.text[:200]}",
                  file=sys.stderr)
            return None
        result = resp.json()
        text = "".join(b.get("text", "") for b in result.get("content", []))
    except Exception as e:
        print(f"  ✗ DeepSeek API error: {e}", file=sys.stderr)
        return None

    return _parse_llm_json(text)


def _parse_llm_json(text: str) -> Optional[dict]:
    """Extract JSON object from LLM response text."""
    # Remove markdown code fences if present
    for fence in ["```json", "```"]:
        text = text.replace(fence, "")
    text = text.strip()

    # Find the outermost JSON braces
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        print(f"  ✗ No JSON found in response: {text[:300]}", file=sys.stderr)
        return None

    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON parse error: {e}", file=sys.stderr)
        print(f"  Raw: {text[start:end+1][:500]}", file=sys.stderr)
        return None


# ─── Validation ─────────────────────────────────────────────────────────────

def validate_template(template: dict) -> Tuple[bool, List[str]]:
    """Validate a generated template against the instrument catalog and schema.

    Returns (is_valid, list_of_errors).
    """
    errors = []

    # Required top-level keys
    for key in ["template_name", "room", "doorways", "furniture", "instrument_slots"]:
        if key not in template:
            errors.append(f"Missing required key: '{key}'")

    # Validate instrument slots
    slots = template.get("instrument_slots", {})
    if not slots:
        errors.append("No instrument_slots defined")
    else:
        for slot_id, slot in slots.items():
            # Allowed instruments
            allowed = slot.get("allowed", [])
            for inst in allowed:
                if inst not in INSTRUMENT_CATALOG:
                    errors.append(
                        f"Slot '{slot_id}': unknown instrument '{inst}'. "
                        f"Available: {list(INSTRUMENT_CATALOG.keys())}")

            # Zone
            zone = slot.get("zone", "")
            if zone not in ("wall", "table_top", "floor", "center"):
                errors.append(f"Slot '{slot_id}': invalid zone '{zone}'")

            # Wall constraint
            if zone in ("wall", "floor") and "wall" not in slot:
                errors.append(
                    f"Slot '{slot_id}': zone='{zone}' requires 'wall' field "
                    f"(one of {WALLS})")
            if "wall" in slot and slot["wall"] not in WALLS:
                errors.append(
                    f"Slot '{slot_id}': invalid wall '{slot['wall']}' "
                    f"(must be one of {WALLS})")

            # Table-top constraint
            if zone == "table_top" and "on_furniture" not in slot:
                errors.append(
                    f"Slot '{slot_id}': zone='table_top' requires "
                    "'on_furniture' field")

            # Check on_furniture references exist
            if "on_furniture" in slot:
                furn = slot["on_furniture"]
                furniture = template.get("furniture", {})
                if furn not in furniture:
                    errors.append(
                        f"Slot '{slot_id}': on_furniture='{furn}' not found "
                        f"in furniture dict. Available: {list(furniture.keys())}")

            # Count
            count = slot.get("count", 0)
            if isinstance(count, (int, float)):
                if count < 0:
                    errors.append(f"Slot '{slot_id}': count must be >= 0")
            elif isinstance(count, list):
                if len(count) != 2 or count[0] > count[1] or count[0] < 0:
                    errors.append(f"Slot '{slot_id}': count list must be [min, max]")

    # Validate furniture
    furniture = template.get("furniture", {})
    for furn_id, furn in furniture.items():
        ck = furn.get("catalog_key", "")
        if ck not in FURNITURE_CATALOG:
            errors.append(
                f"Furniture '{furn_id}': unknown catalog_key '{ck}'. "
                f"Available: {list(FURNITURE_CATALOG.keys())}")

    return len(errors) == 0, errors


# ─── Main API ────────────────────────────────────────────────────────────────

def generate_template(description: str,
                      template_name: Optional[str] = None,
                      use_openai: bool = True,
                      verbose: bool = True) -> Optional[dict]:
    """Generate ONE layout template from a natural language description.

    Returns a validated template dict ready for LAYOUT_TEMPLATES, or None.
    """
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(description, template_name)

    if verbose:
        llm_name = "GPT-4o" if use_openai else "DeepSeek"
        print(f"  Generating layout via {llm_name}...", flush=True)
        print(f"  Description: \"{description[:120]}{'...' if len(description)>120 else ''}\"",
              flush=True)

    template = call_llm(system_prompt, user_prompt, use_openai=use_openai)
    if template is None:
        return None

    # Validate
    is_valid, errors = validate_template(template)
    if not is_valid:
        if verbose:
            print(f"  ⚠  Validation failed ({len(errors)} errors):", file=sys.stderr)
            for e in errors:
                print(f"     - {e}", file=sys.stderr)

        # Try to fix with a second LLM call
        if verbose:
            print(f"  Retrying with error feedback...", flush=True)
        fix_prompt = (
            f"The previous JSON had validation errors:\n"
            + "\n".join(f"  - {e}" for e in errors)
            + f"\n\nPlease output the CORRECTED JSON. The original was:\n"
            + json.dumps(template, indent=2)
        )
        template = call_llm(system_prompt, fix_prompt, use_openai=use_openai)
        if template is None:
            return None
        is_valid, errors = validate_template(template)
        if not is_valid:
            if verbose:
                print(f"  ✗ Still invalid after retry ({len(errors)} errors):",
                      file=sys.stderr)
                for e in errors:
                    print(f"     - {e}", file=sys.stderr)
            return None

    name = template.get("template_name", "unnamed")
    n_slots = len(template.get("instrument_slots", {}))
    n_furn = len(template.get("furniture", {}))
    if verbose:
        print(f"  ✓ Generated: '{name}' ({n_slots} slots, {n_furn} furniture)",
              flush=True)
        print(f"  Description: {template.get('description', 'N/A')}", flush=True)

    return template


def generate_variants(description: str, count: int = 3,
                      base_name: Optional[str] = None,
                      use_openai: bool = True,
                      verbose: bool = True) -> List[dict]:
    """Generate multiple layout variants from one description.

    Each variant is an independent LLM call (non-deterministic), producing
    different slot configurations from the same natural language input.
    """
    variants = []
    for i in range(count):
        if verbose:
            print(f"\n--- Variant {i+1}/{count} ---", flush=True)
        vname = f"{base_name}_v{i+1}" if base_name else None
        template = generate_template(
            description,
            template_name=vname,
            use_openai=use_openai,
            verbose=verbose,
        )
        if template:
            variants.append(template)
        elif verbose:
            print(f"  ⚠  Variant {i+1} failed, skipping", file=sys.stderr)
    return variants


def save_templates(templates: List[dict], filepath: Optional[str] = None) -> str:
    """Save generated templates to a JSON file or return as string.

    If filepath is given, saves to that file.
    Always returns the JSON string.
    """
    if filepath is None:
        filepath = str(Path(__file__).parent / "generated_templates.json")

    # Load existing if any
    existing = {}
    if os.path.exists(filepath):
        with open(filepath) as f:
            existing = json.load(f)

    for t in templates:
        name = t["template_name"]
        existing[name] = {"description": t["description"], **{
            k: v for k, v in t.items()
            if k not in ("template_name", "description")
        }}

    with open(filepath, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    return json.dumps(existing, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="LLM-Assisted BioVLN Layout Template Generation")
    ap.add_argument("--desc", type=str, default=None,
                    help="Natural language description of the lab layout")
    ap.add_argument("--file", type=str, default=None,
                    help="Read description from file")
    ap.add_argument("--name", type=str, default=None,
                    help="Template name (snake_case)")
    ap.add_argument("--variants", type=int, default=1,
                    help="Number of variants to generate (default: 1)")
    ap.add_argument("--llm", type=str, default="openai",
                    choices=["openai", "deepseek"],
                    help="LLM backend (default: openai)")
    ap.add_argument("--save", action="store_true",
                    help="Save to generated_templates.json")
    ap.add_argument("--output", type=str, default=None,
                    help="Custom output file path")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress verbose output")

    args = ap.parse_args()

    # Get description
    desc = args.desc
    if args.file:
        with open(args.file) as f:
            desc = f.read().strip()

    if not desc:
        print("Error: need --desc or --file", file=sys.stderr)
        print("Example: python generate_templates_llm.py --desc "
              "'A molecular biology lab with PCR workstation on the east bench'",
              file=sys.stderr)
        sys.exit(1)

    use_openai = (args.llm == "openai")
    verbose = not args.quiet

    print(f"{'='*60}")
    print(f"  BioVLN LLM Layout Generator ({args.llm.upper()})")
    print(f"{'='*60}")

    templates = generate_variants(
        desc,
        count=args.variants,
        base_name=args.name,
        use_openai=use_openai,
        verbose=verbose,
    )

    if not templates:
        print("\n✗ No valid templates generated.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Generated {len(templates)} valid template(s)")
    print(f"{'='*60}")

    template_json = json.dumps(
        templates if len(templates) > 1 else templates[0],
        indent=2, ensure_ascii=False,
    )
    print(f"\n{template_json}")

    if args.save:
        out_path = args.output
        output = save_templates(templates, out_path)
        path = out_path or str(Path(__file__).parent / "generated_templates.json")
        if not args.quiet:
            print(f"\n✓ Saved to {path}")

    if not args.quiet:
        print(f"\nTo use in config.py, copy the template into LAYOUT_TEMPLATES.")
