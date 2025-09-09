import os
import numpy as np
from typing import List, Dict, Any
from remi_z import Bar, MultiTrack
from pydub import AudioSegment
import subprocess
import contextlib
import fluidsynth
from core.hmm_export import *


class Tab:
    """
    Minimal Tab representation for exporting.
    Matrix shape: 6 x n_positions, rows 0..5 map to strings low E..high e.
    Values: -1 means no note, otherwise fret number (>=0).
    """

    def __init__(self, n_positions: int = 8, bar_data: Any = None, original_onsets: Any = None) -> None:
        # Matrix rows 0..5 map to strings high e..low E (aligned with GATab)
        if bar_data is not None:
            n_positions = len(bar_data)
            self.matrix = np.full((6, n_positions), -1, dtype=int)
            for pos, chord in enumerate(bar_data):
                for string_idx, fret in enumerate(chord):
                    if pos < self.matrix.shape[1] and string_idx < self.matrix.shape[0]:
                        self.matrix[string_idx, pos] = int(fret)
        else:
            self.matrix = np.full((6, n_positions), -1, dtype=int)
        self.chord_dict: Dict[int, str] = {}
        self.original_onsets = original_onsets if original_onsets is not None else []
        self.melody_positions = set()

    def __str__(self) -> str:
        lines = []
        for s in range(self.matrix.shape[0]):
            line = ' '.join([
                '--' if self.matrix[s, p] == -1 else str(int(self.matrix[s, p])).rjust(2)
                for p in range(self.matrix.shape[1])
            ])
            lines.append(line)
        return '\n'.join(lines)

    def add_note(self, position_id: int, string_id: int, fret: int) -> None:
        if not (1 <= string_id <= 6):
            raise ValueError('string_id must be in [1, 6]')
        if not (0 <= position_id < self.matrix.shape[1]):
            raise ValueError(f'position_id must be in [0, {self.matrix.shape[1]-1}]')
        self.matrix[string_id - 1, position_id] = int(fret)

    def add_chord(self, position_id: int, chord_name: str) -> None:
        if not (0 <= position_id < self.matrix.shape[1]):
            raise ValueError(f'position_id must be in [0, {self.matrix.shape[1]-1}]')
        self.chord_dict[position_id] = chord_name

    def add_chord_info(self, position_id: int, chord_name: str) -> None:
        # Alias for GA compatibility
        self.add_chord(position_id, chord_name)

    def add_melody_position(self, position_id: int) -> None:
        if not (0 <= position_id < self.matrix.shape[1]):
            raise ValueError(f'position_id must be in [0, {self.matrix.shape[1]-1}]')
        self.melody_positions.add(position_id)

    @property
    def bar_data(self) -> List[List[int]]:
        result: List[List[int]] = []
        for pos in range(self.matrix.shape[1]):
            chord: List[int] = []
            for string_idx in range(self.matrix.shape[0]):
                chord.append(int(self.matrix[string_idx, pos]))
            result.append(chord)
        return result

    def convert_to_bar(self, guitar: Any = None, resolution: int = 16) -> Bar:
        """
        Convert the Tab to a Bar object.
        - Distribute positions evenly across a full 4/4 bar (no inter-bar gaps).
        - If a Guitar is provided (core.utils.GAutils.Guitar), use its fretboard mapping.
        - Otherwise, assume standard tuning high e..low E: [64, 59, 55, 50, 45, 40].
        """
        open_pitches = [64, 59, 55, 50, 45, 40]
        notes: Dict[int, List[List[int]]] = {}
        n_positions = self.matrix.shape[1]
        for p_pos in range(n_positions):
            for s in range(6):  # 0..5 = highE..lowE
                fret = int(self.matrix[s, p_pos])
                if fret >= 0:
                    pitch = int(open_pitches[s] + fret)
                    dur = 6
                    onset = int(p_pos * dur)
                    velocity = 96
                    note = [pitch, dur, velocity]
                    notes.setdefault(onset, []).append(note)
        time_signature = (4, 4)
        tempo = 120
        return Bar(id=-1, notes_of_insts={0: notes}, time_signature=time_signature, tempo=tempo)


class TabSeq:
    def __init__(self, tab_list: List[Tab] = None, tab_per_row: int = 2) -> None:
        self.tab_list = tab_list if tab_list is not None else []
        self.tab_per_row = tab_per_row

    def __str__(self) -> str:
        tab_lines_list = [str(tab).split('\n') for tab in self.tab_list]
        tab_height = len(tab_lines_list[0]) if tab_lines_list else 0
        lines: List[str] = []
        tabs_per_row = self.tab_per_row
        for i in range(0, len(tab_lines_list), tabs_per_row):
            row_tabs = tab_lines_list[i:i + tabs_per_row]
            tab_objs = self.tab_list[i:i + tabs_per_row]
            while len(row_tabs) < tabs_per_row:
                row_tabs.append([' ' * (len(row_tabs[0][0]) if row_tabs else 10)] * tab_height)
                tab_objs.append(None)
            chord_name_lines: List[str] = []
            for tab, tab_lines in zip(tab_objs, row_tabs):
                chord_line = [' ' for _ in range(len(tab_lines[0]))]
                if tab is not None and hasattr(tab, 'chord_dict') and tab.chord_dict:
                    n_positions = len(tab_lines[0].split())
                    sorted_positions = sorted(tab.chord_dict.keys())
                    if len(sorted_positions) > 0:
                        first_pos = sorted_positions[0]
                        chord1 = tab.chord_dict[first_pos]
                        chord_line_pos = first_pos * 3
                        chord_line[chord_line_pos:chord_line_pos+len(chord1)] = chord1
                    if len(sorted_positions) > 1:
                        second_pos = sorted_positions[1]
                        chord2 = tab.chord_dict[second_pos]
                        chord_line_pos = second_pos * 3
                        chord_line[chord_line_pos:chord_line_pos+len(chord2)] = chord2
                chord_name_lines.append(''.join(chord_line))
            lines.append('   '.join(chord_name_lines))
            for line_idx in range(tab_height):
                row_line = '   '.join(tab[line_idx] for tab in row_tabs)
                lines.append(row_line)
            lines.append('')
        return '\n'.join(lines)

    def save_to_file(self, filename: str) -> None:
        with open(filename, 'w') as f:
            f.write(str(self))

    def add_tab(self, tab: Tab) -> None:
        self.tab_list.append(tab)

    def convert_to_note_seq(self) -> MultiTrack:
        # Backward-compatible wrapper using default resolution and internal tuning
        bars = [tab.convert_to_bar(guitar=None, resolution=16) for tab in self.tab_list]
        return MultiTrack.from_bars(bars)

    def convert_to_multitrack(self, guitar: Any, tempo: int = 120, time_signature: Any = (4, 4), resolution: int = 16) -> MultiTrack:
        bars = []
        for tab in self.tab_list:
            bar = tab.convert_to_bar(guitar=guitar, resolution=resolution)
            bar.tempo = tempo
            bar.time_signature = time_signature
            bars.append(bar)
        return MultiTrack.from_bars(bars)


def path_to_tabseq(path: List[Dict[str, Any]], positions_per_bar: int = 8) -> TabSeq:
    tabs: List[Tab] = []
    for start_idx in range(0, len(path), positions_per_bar):
        segment = path[start_idx:start_idx + positions_per_bar]
        if not segment:
            continue
        tab = Tab(n_positions=len(segment))
        for local_pos, form in enumerate(segment):
            fret_config = form.get('fret_config', {})
            for string_idx in range(6):
                # fret_config is lowE to highE
                fret = fret_config.get(5-string_idx, fret_config.get(str(5-string_idx), -1))
                if isinstance(fret, (int, np.integer)) and fret >= 0:
                    tab.add_note(local_pos, string_idx+1, int(fret))
        tabs.append(tab)
    return TabSeq(tabs)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def midi_to_wav(midi_fp: str, sf_path: str, wav_fp: str, verbose: bool = True) -> None:
    cmd = [
        'fluidsynth',
        '-ni',
        sf_path,
        midi_fp,
        '-F', wav_fp,
        '-r', '44100',
    ]
    if verbose:
        subprocess.run(cmd, check=True)
    else:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def post_process_wav(wav_fp: str, silence_thresh_db: float = -40.0, padding_ms: int = 200, target_dbfs: float = -0.1) -> None:
    audio = AudioSegment.from_wav(wav_fp)
    change_in_dBFS = target_dbfs - audio.max_dBFS
    normalized = audio.apply_gain(change_in_dBFS)
    # trim
    start_trim = _detect_leading_silence(normalized, silence_thresh_db)
    end_trim = _detect_leading_silence(normalized.reverse(), silence_thresh_db)
    duration = len(normalized)
    trimmed = normalized[start_trim:duration - end_trim + padding_ms]
    trimmed.export(wav_fp, format='wav')


def _detect_leading_silence(sound: AudioSegment, silence_threshold: float = -40.0, chunk_size: int = 10) -> int:
    trim_ms = 0
    while trim_ms < len(sound) and sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold:
        trim_ms += chunk_size
    return trim_ms


def export_hmm_path(path: List[Dict[str, Any]], song_name: str = 'arrangement', positions_per_bar: int = 8, tempo: int = 90,
                    output_dir: str = 'outputs', sf2_path: str = 'resources/Tyros Nylon.sf2',
                    export_separate: bool = True, verbose: bool = True) -> Dict[str, str]:
    """
    End-to-end export following core utils pipeline:
    - Convert path -> TabSeq
    - Save .tab text
    - Convert to MultiTrack, set tempo, save .midi
    - Render .wav via fluidsynth and post-process
    - If export_separate=True, files are placed under output_dir/song_name/
    Returns dict of file paths.
    """
    # Destination directory per arrangement
    dest_dir = os.path.join(output_dir, song_name) if export_separate else output_dir
    ensure_dir(dest_dir)

    tabseq = path_to_tabseq(path, positions_per_bar=positions_per_bar)

    tab_fp = os.path.join(dest_dir, f'{song_name}.tab')
    with open(tab_fp, 'w') as f:
        f.write(str(tabseq))

    mt = tabseq.convert_to_note_seq()
    mt.set_tempo(tempo)

    midi_fp = os.path.join(dest_dir, f'{song_name}.midi')
    if verbose:
        mt.to_midi(midi_fp)
    else:
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                mt.to_midi(midi_fp)

    wav_fp = os.path.join(dest_dir, f'{song_name}.wav')
    midi_to_wav(midi_fp, sf2_path, wav_fp, verbose=verbose)
    post_process_wav(wav_fp)

    return {
        'tab_fp': tab_fp,
        'midi_fp': midi_fp,
        'wav_fp': wav_fp,
    }


