import json
from typing import List, Dict, Any


OPEN_STRINGS = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, E4 (string indices 0..5)


def convert_tab_to_fret_config(tab: List[Any]) -> Dict[str, int]:
    """
    Convert a tab array into a fret configuration mapping.
    - Keys are string indices as strings: "0".."5" (0 = low E)
    - Values are fret numbers; -1 for muted strings
    """
    fret_config: Dict[str, int] = {}
    for i, fret in enumerate(tab):
        if fret == "x":
            fret_config[str(i)] = -1
        else:
            fret_config[str(i)] = int(fret)
    return fret_config


def calculate_pitches_from_tab(tab: List[Any]) -> List[int]:
    """
    Calculate MIDI pitches from a tab using OPEN_STRINGS.
    """
    pitches: List[int] = []
    for string_idx, fret in enumerate(tab):
        if fret != "x":
            pitches.append(OPEN_STRINGS[string_idx] + int(fret))
    return sorted(pitches)


def calculate_form_properties(tab: List[Any], fingering: List[int]) -> Dict[str, Any]:
    """
    Compute index_pos, width, fingers, finger_used, and string indices (0-based) from tab/fingering.
    - index_pos: minimum pressed fret (>0). 0 for open-only shapes.
    - width: span of pressed frets (max - min + 1), 0 if no pressed frets.
    - fingers: number of distinct fingers used (>0 in fingering array).
    - finger_used: sorted distinct finger ids used (>0).
    - string: 0-based indices of non-muted strings.
    """
    pressed_frets: List[int] = []
    fingers_used: List[int] = []
    strings_used: List[int] = []

    for i, (fret, finger) in enumerate(zip(tab, fingering)):
        if fret != "x":
            strings_used.append(i)
            if fret != 0:
                pressed_frets.append(int(fret))
            if isinstance(finger, int) and finger > 0:
                fingers_used.append(int(finger))

    if pressed_frets:
        min_fret = min(pressed_frets)
        max_fret = max(pressed_frets)
        index_pos = min_fret
        width = max_fret - min_fret + 1
    else:
        index_pos = 0
        width = 0

    finger_used_sorted = sorted(set(fingers_used))

    return {
        "index_pos": index_pos,
        "width": width,
        "fingers": len(finger_used_sorted),
        "finger_used": finger_used_sorted,
        "string": strings_used,
    }


def convert_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    tab: List[Any] = entry.get("tab", [])
    fingering: List[int] = entry.get("fingering", [0] * len(tab))

    # Prefer pitches from the entry (if present), otherwise compute
    if "pitches" in entry and isinstance(entry["pitches"], list):
        pitches = sorted(int(p) for p in entry["pitches"])  # ensure ints and sorted
    else:
        pitches = calculate_pitches_from_tab(tab)

    fret_config = convert_tab_to_fret_config(tab)

    # If index_pos exists in entry, keep it to stay faithful to source; else compute
    if "index_pos" in entry:
        index_pos = int(entry["index_pos"])  # type: ignore[arg-type]
        props = calculate_form_properties(tab, fingering)
        props["index_pos"] = index_pos
    else:
        props = calculate_form_properties(tab, fingering)

    return {
        "pitches": pitches,
        "fret_config": {str(k): int(v) for k, v in fret_config.items()},
        "index_pos": props["index_pos"],
        "width": props["width"],
        "fingers": props["fingers"],
        "finger_used": props["finger_used"],
        "string": props["string"],
    }


def convert_file(input_file: str, output_file: str) -> List[Dict[str, Any]]:
    """
    Convert a JSONL (or JSON array) file into the target multi-forms JSON format.
    """
    with open(input_file, "r") as f:
        content = f.read()

    forms: List[Dict[str, Any]] = []

    if content.strip().startswith("["):
        data = json.loads(content)
        for entry in data:
            forms.append(convert_entry(entry))
    else:
        for line in content.strip().split("\n"):
            if line.strip():
                entry = json.loads(line)
                forms.append(convert_entry(entry))

    with open(output_file, "w") as f:
        json.dump(forms, f, indent=2)

    return forms


if __name__ == "__main__":
    # Default paths within this repo
    input_path = "possible_positions.jsonl"
    output_path = "guitar_forms_from_jsonl.json"
    convert_file(input_path, output_path)


