import json
from typing import List, Dict, Any

OPEN_STRINGS = [40, 45, 50, 55, 59, 64]

def convert_tab_to_fret_config(tab: List[Any]) -> Dict[str, int]:
	fret_config: Dict[str, int] = {}
	for i, fret in enumerate(tab):
		if fret == "x":
			fret_config[str(i)] = -1
		else:
			fret_config[str(i)] = int(fret)
	return fret_config


def calculate_pitches_from_tab(tab: List[Any]) -> List[int]:
	pitches: List[int] = []
	for string_idx, fret in enumerate(tab):
		if fret != "x":
			pitches.append(OPEN_STRINGS[string_idx] + int(fret))
	return sorted(pitches)


def calculate_form_properties(tab: List[Any], fingering: List[int]) -> Dict[str, Any]:
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
	if "pitches" in entry and isinstance(entry["pitches"], list):
		pitches = sorted(int(p) for p in entry["pitches"])  # ensure ints and sorted
	else:
		pitches = calculate_pitches_from_tab(tab)
	fret_config = convert_tab_to_fret_config(tab)
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
