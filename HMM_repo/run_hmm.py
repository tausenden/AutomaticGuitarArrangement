import os
from core.hmm_repro import HMMrepro
from core.hmm_export import export_hmm_bars
from core.utils.hmm_utils import get_notes_grouped_by_bars

# Configuration (edit these as needed)
INPUT_DIR = '../song_midis'
OUTPUT_DIR = '../arranged_songs_hmm'
FORMS_FILES = ['./states/guitar_forms_CAGED_2.json']
TEMPO = 100
MIN_TAB_POSITIONS = 8


def arrange_midi_file(hmm: HMMrepro, midi_file_path: str, output_dir: str, tempo: int = 100) -> None:
	bars_notes = get_notes_grouped_by_bars(midi_file_path)
	bars_paths = []
	for bar_idx, bar_positions in enumerate(bars_notes):
		if not bar_positions:
			bars_paths.append([])
			continue
		segment_path = hmm.viterbi(bar_positions)
		if segment_path:
			bars_paths.append(segment_path)
		else:
			bars_paths.append([])
	path_bars = bars_paths
	if any(len(seg) > 0 for seg in path_bars):
		song_name = os.path.splitext(os.path.basename(midi_file_path))[0]
		export_hmm_bars(path_bars, song_name=song_name, tempo=tempo, output_dir=output_dir, min_positions_per_bar=MIN_TAB_POSITIONS)


if __name__ == '__main__':
	os.makedirs(OUTPUT_DIR, exist_ok=True)

	hmm = HMMrepro(forms_files=FORMS_FILES)

	midi_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.mid', '.midi'))]
	midi_files.sort()
	for filename in midi_files:
		midi_file_path = os.path.join(INPUT_DIR, filename)
		print(f'Processing {midi_file_path} ...')
		try:
			arrange_midi_file(hmm, midi_file_path, OUTPUT_DIR, tempo=TEMPO)
		except Exception as e:
			print(f'Failed on {filename}: {e}')
