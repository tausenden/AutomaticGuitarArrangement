from remi_z import MultiTrack

def get_all_notes_from_midi(midi_file_path):
	"""
	Extract all notes from a MIDI file as a list of lists with pitch values.
	"""
	from remi_z import MultiTrack
	mt = MultiTrack.from_midi(midi_file_path)
	all_positions = []
	for bar in mt.bars:
		position_notes = {}
		all_notes = bar.get_all_notes(include_drum=False)
		for note in all_notes:
			pos = note.onset
			if pos not in position_notes:
				position_notes[pos] = []
			position_notes[pos].append(note.pitch)
			position_notes[pos] = list(set(position_notes[pos]))
		sorted_positions = sorted(position_notes.keys())
		for pos in sorted_positions:
			all_positions.append(position_notes[pos])
	return all_positions


def get_notes_grouped_by_bars(midi_file_path):
	"""
	Extract notes grouped by bar. Returns List[List[List[int]]]
	where outer list indexes bars, inner list indexes positions within a bar,
	and each position is a list of MIDI pitches (deduplicated).
	"""
	mt = MultiTrack.from_midi(midi_file_path)
	bars_notes = []
	for bar in mt.bars:
		position_notes = {}
		all_notes = bar.get_all_notes(include_drum=False)
		for note in all_notes:
			pos = note.onset
			if pos not in position_notes:
				position_notes[pos] = []
			position_notes[pos].append(note.pitch)
			position_notes[pos] = sorted(list(set(position_notes[pos])))
		sorted_positions = sorted(position_notes.keys())
		bar_positions = [position_notes[pos] for pos in sorted_positions]
		bars_notes.append(bar_positions)
	return bars_notes


def _flatten_bars_to_sequence(bars_notes):
	"""Flatten a list of bars (each a list of positions) into a single sequence of positions."""
	flat = []
	for bar in bars_notes:
		flat.extend(bar)
	return flat
