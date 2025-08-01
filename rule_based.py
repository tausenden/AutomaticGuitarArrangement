'''
这个文件用来描述Rule-Based Guitar Arranger的伪代码
'''
from remi_z import MultiTrack, NoteSeq, ChordSeq, midi_pitch_to_note_name, Note
from typing import List
from fretboard_util import Fretboard
from tab_util import Chart, Tab, TabSeq
import fluidsynth
import subprocess
from pydub import AudioSegment


def main():
    arranger = ArrangerSystem()
    midi_fp = 'misc/caihong-4bar.midi'
    # midi_fp = 'misc/canon_in_D.mid'
    arranger.arrange_song_from_midi(midi_fp)


class ArrangerSystem:
    def __init__(self):
        self.mt = None
        self.voicer = Voicer()
        self.arpeggiator = Arpeggiator()

    def arrange_song_from_midi(self, midi_fp):
        song_name = midi_fp.split('/')[-1].split('.')[0]
        print(f'Arranging song: {song_name}')

        self.mt = MultiTrack.from_midi(midi_fp)[:8] # [0:1]
        self.mt.quantize_to_16th()
        # self.mt.shift_pitch(-5)
        self.mt.shift_pitch(2)

        # Prepare model inputs
        notes_of_bars = self.get_all_notes()
        melody = self.extract_melody()
        chord_of_bars = self.extract_chord()

        # Get chart sequence (left hand modeling)
        chart_seq = self.voicer.generate_chart_sequence_for_song(melody, chord_of_bars)
        chart_fp = f'{song_name}.chart'
        print(f'Saving chart sequence to {chart_fp}')
        self.voicer.save_chart_sequence_to_file(chart_seq, chart_fp)

        # Tab generation given charts (right hand modeling)
        song_tab = self.arpeggiator.arpeggiate_a_song(melody, chart_seq, notes_of_bars)

        # Save the tab to file
        tab_fp = f'{song_name}.tab'
        print(f'Saving tab to {tab_fp}')
        song_tab.save_to_file(tab_fp)

        # Convert to note sequence
        mt = song_tab.convert_to_note_seq()
        # more_accurate_note_seq = duration_renderer(out_note_seq)

        # Set tempo
        mt.set_tempo(90)

        # Save the note sequence to MIDI
        # midi_fp = 'test_out.mid'
        midi_fp = f'{song_name}.midi'
        print(f'Saving MIDI to {midi_fp}')
        mt.to_midi(midi_fp)

        # Synthesize the MIDI to WAV
        sf_path = 'resources/Tyros Nylon.sf2'
        audio_fp = f'{song_name}.wav'
        # wav_fp = 'test_out.wav'
        print(f'Synthesizing MIDI to WAV: {audio_fp}')
        self.midi_to_wav(midi_fp, sf_path, audio_fp)
        self.post_process_wav(audio_fp, silence_thresh_db=-40.0, padding_ms=200, target_dbfs=-0.1)

    
    def midi_to_note_seq(self, midi_fp):
        '''
        Convert MIDI file to note sequence
        '''
        mt = MultiTrack.from_midi(midi_fp)
        mt.quantize_to_16th()
        note_seq_per_bar = mt.get_all_notes_by_bar()
        mt.get_note_list

        melody = mt.get_melody('hi_track')

        return note_seq_per_bar
    
    def extract_melody(self):
        melody = self.mt.get_melody('hi_note')
        ret = []
        for melody_of_bar in melody:
            ret.append(NoteSeq(melody_of_bar))
        return ret

    def get_all_notes(self):
        notes_of_bars = self.mt.get_all_notes_by_bar()
        ret = []
        for notes_of_bar in notes_of_bars:
            ret.append(NoteSeq(notes_of_bar))
        return ret
    
    def extract_chord(self):
        chords = [bar.get_chord() for bar in self.mt]
        ret = []
        for chord_of_bar in chords:
            ret.append(ChordSeq(chord_of_bar))
        return ret
    
    def midi_to_wav(self, midi_fp, sf_path, wav_fp):
        cmd = [
            "fluidsynth",
            "-ni",
            sf_path,
            midi_fp,
            "-F", wav_fp,
            "-r", "44100"
        ]
        subprocess.run(cmd, check=True)

    def post_process_wav(self, wav_fp, silence_thresh_db=-40.0, padding_ms=200, target_dbfs=-0.1):
        audio = AudioSegment.from_wav(wav_fp)

        # Step 1: Normalize to target dBFS
        change_in_dBFS = target_dbfs - audio.max_dBFS
        normalized = audio.apply_gain(change_in_dBFS)

        # Step 2: Trim silence after normalization
        start_trim = self.detect_leading_silence(normalized, silence_thresh_db)
        end_trim = self.detect_leading_silence(normalized.reverse(), silence_thresh_db)
        duration = len(normalized)
        trimmed = normalized[start_trim:duration - end_trim + padding_ms]

        # Export
        trimmed.export(wav_fp, format="wav")


    def detect_leading_silence(self, sound, silence_threshold=-40.0, chunk_size=10):
        trim_ms = 0
        while trim_ms < len(sound) and sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold:
            trim_ms += chunk_size
        return trim_ms

class Block:
    def __init__(self, chord:tuple, melody:NoteSeq):
        self.chord = chord
        self.melody = melody

    def __str__(self):
        return f'Chord: {self.chord}, Melody: {self.melody}'
    
    def __repr__(self):
        return self.__str__()


class Voicer:
    def __init__(self):
        self.fretboard = Fretboard()

    def generate_chart_candidate_for_block(self, block:Block):
        '''
        生成chart候选 for a block
        Block means all melody covered by a same chord
        or half of a bar
        which is shorter.

        Input:
        - melody_notes: list of notes in the melody, e.g. ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]
        - chord: a chord string, e.g. "Cmaj7"

        Return:
        - chart_candidate: a list of chart candidates, e.g., [chart1, chart2, chart3]
        Each chart indicate a position on guitar neck and all string-fret that contains melody notes and all chord notes.

        '''
        # Find all possible positions to fret these melody notes
        melody = block.melody.get_note_name_list()
        possible_positions = self.fretboard.find_all_playable_position_of_note_set(melody)
        melody_pitch_range = block.melody.get_pitch_range()
        chord_upper_pitch_limit = melody_pitch_range[0]
        lowest_melody_note = midi_pitch_to_note_name(chord_upper_pitch_limit)

        # If no possible position, raise warning
        # print(len(possible_positions))
        if len(possible_positions) == 0:
            print(f'Warning: No possible position found for melody {melody} with chord {block.chord}.')
            return []

        # For each position
        ret = []
        for position in possible_positions:
            # Find all chord note in that position (try press)
            chord_str = f'{block.chord[0]}{block.chord[1]}'
            chord_note_sfs = self.fretboard.press_chord(chord_str, 
                                                        position=position, 
                                                        string_press_once=False, 
                                                        enforce_root=True,
                                                        closed=False,
                                                        force_highest_pitch=lowest_melody_note,
                                                        )
            
            # If fail to press this chord, skip this position
            thres = 1 # at least we want 2 useable strings in the chord note
            chord_strings = set([sf[0] for sf in chord_note_sfs])
            if len(chord_strings) < thres:
                continue

            # Draw a chart
            chord_name = f'{block.chord[0]}{block.chord[1]}'
            chart = Chart(string_fret_list=chord_note_sfs, chord_name=chord_name)

            # Press the melody notes in that position
            melody_note_sfs = self.fretboard.press_note_set(note_set=melody, lowest_fret=position)

            # Add the melody notes to the chart
            melody_str = ' '.join(melody)
            chart.fret_more_note(melody_note_sfs, melody_list=melody_str)
            ret.append(chart)

        # If no chart candidate, raise warning
        if len(ret) == 0:
            print(f'Warning: No chart candidate found for melody {block.melody} with chord {block.chord}.')
            print('possible positions for melody only:', possible_positions)
            return []

        return ret

    def generate_chart_candidate_sequence_for_bar(self, melody_of_bar:NoteSeq, chord_of_bar:ChordSeq):
        '''
        Return a chart candidate sequence
        '''
        # Break melody and chord into blocks
        # Here we assume half a bar is a block
        melody_of_block_1 = NoteSeq([note for note in melody_of_bar.notes if note.onset<24])
        chord_of_block_1 = chord_of_bar.chord_list[0]
        melody_of_block_2 = NoteSeq([note for note in melody_of_bar.notes if note.onset>=24])
        chord_of_block_2 = chord_of_bar.chord_list[1]
        block_1 = Block(chord = chord_of_block_1, melody=melody_of_block_1)
        block_2 = Block(chord=chord_of_block_2, melody=melody_of_block_2)
        blocks = [block_1, block_2]

        ret = []
        for block in blocks:
            chart_candidates_of_block = self.generate_chart_candidate_for_block(block)
            ret.append(chart_candidates_of_block)    

        return ret

    def find_best_chart_path(self, chart_candidates_seq):
        '''
        Find the best path for the song
        That minimize the left hand position movement
        Uses Dijkstra's algorithm to find the path with minimal sum of avg_fret differences.
        '''
        import heapq
        if not chart_candidates_seq or not all(chart_candidates_seq):
            return []
        n_layers = len(chart_candidates_seq)
        # Each node is (layer_idx, chart_idx)
        # Dijkstra: (total_cost, layer_idx, chart_idx, path_so_far)
        heap = []
        for idx, chart in enumerate(chart_candidates_seq[0]):
            heapq.heappush(heap, (0, 0, idx, [idx]))  # cost, layer, chart_idx, path (as indices)
        # best_cost[layer][chart_idx] = cost
        best_cost = [{} for _ in range(n_layers)]
        for idx in range(len(chart_candidates_seq[0])):
            best_cost[0][idx] = 0
        while heap:
            cost, layer, idx, path = heapq.heappop(heap)
            if layer == n_layers - 1:
                # Reached last layer, connect to end node (cost 0)
                return [chart_candidates_seq[i][j] for i, j in enumerate(path)]
            current_chart = chart_candidates_seq[layer][idx]
            for next_idx, next_chart in enumerate(chart_candidates_seq[layer + 1]):
                edge_cost = abs(current_chart.avg_fret - next_chart.avg_fret)
                new_cost = cost + edge_cost
                if next_idx not in best_cost[layer + 1] or new_cost < best_cost[layer + 1][next_idx]:
                    best_cost[layer + 1][next_idx] = new_cost
                    heapq.heappush(heap, (new_cost, layer + 1, next_idx, path + [next_idx]))
        # If we get here, no path found
        return []

    def generate_chart_sequence_for_song(self, melody_of_bars:List[NoteSeq], chord_progression):
        '''
        Find the best chart sequence for the melody and chord progression of a song.
        That minimize the left-hand position-wise movement on neck

        Implemented by a shortest path algorithm.
        '''
        assert len(melody_of_bars) == len(chord_progression)
        chart_candidates = []
        for melody_of_bar, chord_of_bar in zip(melody_of_bars, chord_progression):
            chart_candidates.extend(self.generate_chart_candidate_sequence_for_bar(melody_of_bar, chord_of_bar))
        
        best_chart_seq = self.find_best_chart_path(chart_candidates)

        return best_chart_seq

    def save_chart_sequence_to_file(self, chart_sequence, filename):
        '''
        Save the chart sequence to a .chart file as plain text, using __str__ of each Chart,
        and including chord/melody info if present.
        '''
        with open(filename, 'w', encoding='utf-8') as f:
            for i, chart in enumerate(chart_sequence):
                f.write(f'Chart {i+1}\n')
                f.write(f'Chord: {chart.chord_name}\n')
                f.write(f'Melody: {chart.melody_list}\n')
                f.write(str(chart))
                f.write('\n')
                f.write('-' * 40 + '\n')


class Arpeggiator:
    '''
    Takes in a sequence of chart, and a note sequence as texture reference,
    Generate tab
    '''
    def __init__(self):
        self.fretboard = Fretboard()
        
    def calculate_groove_for_a_bar(note_seq):
        '''
        groove is represented by 
        onset position of bass note
        onset position of melody note
        counter melody onset position (highest note of filling)
        filling note onset density of each position (except melody and bass onset)
        '''
        pass

    def arpeggiate_a_bar(self, melody, chart_list_of_the_bar:List[Chart], notes_of_the_bar:NoteSeq):
        '''
        这个算法用于为吉他独奏编曲中的“填充声部”（filling）部分分配右手手指（或弦位），以最大程度还原原曲质感。

        首先，从去除主旋律和低音后的原曲 MIDI 中提取每个位置的填充音符，识别出在每个和弦块（chord block）中 
            filling 部分的最高音，并将这些最高音连成一个“filling melody contour”。根据该 contour 的平均音高判断其整体处于高音区还是低音区。

        接着，计算每个非主旋律、非低音位置上的填充音符密度，并取所有位置的中位数作为 density 阈值。

        最后，对于每个 filling 位置，若其密度高于中位数，则分配两个手指进行填充（即两个音）；否则仅分配一个音，
            并根据 filling melody contour 所在音区，选择靠近的高音弦或低音弦来放置音符。

        该策略在尽量还原原曲的同时，控制右手复杂度，并利用简单规则生成合理可演奏的填充纹理。

        '''
        notes_first_half = [note for note in notes_of_the_bar.notes if note.onset < 24]
        notes_second_half = [note for note in notes_of_the_bar.notes if note.onset >= 24]
        melody_first_half = NoteSeq([note for note in melody.notes if note.onset < 24])
        melody_second_half = NoteSeq([note for note in melody.notes if note.onset >= 24])
        assert len(chart_list_of_the_bar) == 2, "There should be exactly 2 charts for a bar."
        assert len(notes_of_the_bar.notes) > 0, "There should be notes in the bar."

        # Feature extraction, getting 

        # 1. Melody note onset position
        melody_onset_positions = [note.onset // 6 for note in melody.notes]

        # 2. Bass note onset position
        bass_note_1_class = chart_list_of_the_bar[0].chord_name[:1] if '#' not in chart_list_of_the_bar[0].chord_name else chart_list_of_the_bar[0].chord_name[:2]
        bass_note_2_class = chart_list_of_the_bar[1].chord_name[:1] if '#' not in chart_list_of_the_bar[1].chord_name else chart_list_of_the_bar[1].chord_name[:2]
        bass_note_1 = get_bass_note_seq(notes_of_the_bar, bass_note_1_class)
        bass_note_2 = get_bass_note_seq(notes_of_the_bar, bass_note_2_class)
        bass_notes = NoteSeq(bass_note_1.notes + bass_note_2.notes)
        bass_onset_positions = [note.onset // 6 for note in bass_notes.notes]

        # 3. Filling note density by position
        fillings = get_filling_notes(notes_of_the_bar, melody, bass_notes)
        fillings_density_by_position = get_filling_note_density_by_position(fillings)

        # 4. Filling sub-melody contour (highest note of filling notes in each onset position)
        sub_melody_contour = get_filling_submelody_contour(fillings)

        # Generate an empty tab
        tab = Tab()

        # Fill melody note to melody note position
        chart_1 = chart_list_of_the_bar[0]
        chart_2 = chart_list_of_the_bar[1]
        psf_1 = self.get_psf_from_chart(melody_first_half, chart_1)
        psf_2 = self.get_psf_from_chart(melody_second_half, chart_2)
        for psf in psf_1:
            pos, string_id, fret, note_name = psf
            tab.add_note(pos, string_id, fret)
        for psf in psf_2:
            pos, string_id, fret, note_name = psf
            tab.add_note(pos, string_id, fret)
        
        # Fill bass note to bass position
        bass_psf_1 = self.get_psf_from_chart(bass_note_1, chart_1)
        bass_psf_2 = self.get_psf_from_chart(bass_note_2, chart_2)
        for psf in bass_psf_1:
            pos, string_id, fret, note_name = psf
            tab.add_note(pos, string_id, fret)
        for psf in bass_psf_2:
            pos, string_id, fret, note_name = psf
            tab.add_note(pos, string_id, fret)

        # Add chord
        chord_1 = chart_1.chord_name
        chord_2 = chart_2.chord_name
        tab.add_chord(0, chord_1)
        tab.add_chord(4, chord_2)

        # Add fills: 
        '''
        这个算法用于为吉他独奏编曲中的“填充声部”（filling）部分分配右手手指（或弦位），以最大程度还原原曲质感。

        首先，从去除主旋律和低音后的原曲 MIDI 中提取每个位置的填充音符，识别出在每个和弦块（chord block）中 
            filling 部分的最高音，并将这些最高音连成一个“filling melody contour”。根据该 contour 的平均音高判断其整体处于高音区还是低音区。

        接着，计算每个非主旋律、非低音位置上的填充音符密度，并取所有位置的中位数作为 density 阈值。

        最后，对于每个 filling 位置，若其密度高于中位数，则分配两个手指进行填充（即两个音）；否则仅分配一个音，
            并根据 filling melody contour 所在音区，选择靠近的高音弦或低音弦来放置音符。

        该策略在尽量还原原曲的同时，控制右手复杂度，并利用简单规则生成合理可演奏的填充纹理。
        '''
        a = 2

        return tab
    
    def get_psf_from_chart(self, melody: 'NoteSeq', chart: 'Chart') -> list:
        """
        Find every position-string-fret pair for each melody note in the chart.

        Args:
            melody: NoteSeq object containing melody notes (each note must have .get_note_name()).
            chart: Chart object containing string-fret mapping for the bar.

        Returns:
            List of tuples (string_id, fret, note_name) for each melody note found in the chart.
        """
        result = []

        for melody_note in melody:
            pos = melody_note.onset // 6  # Convert to 8th note position
            note_name = melody_note.get_note_name()

            # Find the string-fret pair for this melody note in the chart
            sf = chart.get_sf_from_note_name(note_name)
            assert sf is not None, f"Melody note {note_name} not found in chart."
            string_id, fret = sf
            result.append((pos, string_id, fret, note_name))
        return result

    def arpeggiate_a_song(self, melody, chart_list_of_the_song, note_seq_of_the_song) -> TabSeq:
        # Group chart_list by bar
        chart_list_of_bars = []
        chart_list_of_bar = []
        for chart in chart_list_of_the_song:
            if len(chart_list_of_bar) < 2:
                chart_list_of_bar.append(chart)
                if len(chart_list_of_bar) == 2:
                    chart_list_of_bars.append(chart_list_of_bar)
                    chart_list_of_bar = []

        # Call arpeggiate_a_bar function to do the job for each bar.
        tab_of_the_bars = []
        for melody_of_bar, chart_list_of_bar, note_seq_of_bar in zip(melody, chart_list_of_bars, note_seq_of_the_song):
            tab_of_the_bar = self.arpeggiate_a_bar(melody_of_bar, chart_list_of_bar, note_seq_of_bar)
            tab_of_the_bars.append(tab_of_the_bar)
            a = 2
        tab_of_the_song = TabSeq(tab_of_the_bars)
        
        
        return tab_of_the_song


class DurationRenderer:
    '''
    Modify the duration of the system to make them sounds more natural and practical
    '''
    def modify_duration(note_seq):
        return note_seq
    

def get_bass_note_seq(note_seq: 'NoteSeq', chord_root: str) -> 'NoteSeq':
    """
    Given a NoteSeq, find the bass note (lowest pitch) at each onset position
    whose pitch class matches the chord root, and return a new NoteSeq containing
    these bass notes. Only keep bass notes in the most common octave among matched notes.

    Args:
        note_seq: NoteSeq object containing notes (each note must have .onset, .get_note_name(), .pitch).
        chord_root: The root note name of the chord (e.g., "C", "G#", "A").

    Returns:
        NoteSeq containing the bass notes for each onset position, filtered by octave.
    """
    bass_notes = []
    onset_dict = {}
    for note in note_seq.notes:
        onset_dict.setdefault(note.onset, []).append(note)
    for onset, notes_at_onset in onset_dict.items():
        matching_notes = [note for note in notes_at_onset if chord_root in note.get_note_name()]
        if matching_notes:
            bass_note = min(matching_notes, key=lambda n: n.pitch)
            bass_notes.append(bass_note)
    # Filter by most common octave
    if bass_notes:
        # Extract octave from note name (assume last char is octave, e.g. "C3")
        octaves = [int(note.get_note_name()[-1]) for note in bass_notes]
        from collections import Counter
        most_common_octave = Counter(octaves).most_common(1)[0][0]
        bass_notes = [note for note in bass_notes if int(note.get_note_name()[-1]) == most_common_octave]
    return NoteSeq(bass_notes)


def get_filling_notes(note_seq: 'NoteSeq', melody_seq: 'NoteSeq', bass_seq: 'NoteSeq') -> 'NoteSeq':
    """
    Return a NoteSeq containing notes that are neither melody nor bass notes.

    Args:
        note_seq: NoteSeq object containing all notes.
        melody_seq: NoteSeq object containing melody notes.
        bass_seq: NoteSeq object containing bass notes.

    Returns:
        NoteSeq containing only filling notes.
    """
    melody_set = set((note.onset, note.pitch) for note in melody_seq.notes)
    bass_set = set((note.onset, note.pitch) for note in bass_seq.notes)
    filling_notes = [
        note for note in note_seq.notes
        if (note.onset, note.pitch) not in melody_set and (note.onset, note.pitch) not in bass_set
    ]
    return NoteSeq(filling_notes)


def get_filling_note_density_by_position(filling_notes: 'NoteSeq') -> dict:
    """
    Calculate the filling note density for each onset position directly from filling_notes.

    Args:
        filling_notes: NoteSeq object containing only filling notes (each note must have .onset).

    Returns:
        Dictionary mapping onset position (int) to filling note density (int).
    """
    density_by_position = {}
    for note in filling_notes.notes:
        onset = note.onset // 6  # Convert to 8th note position if needed
        density_by_position[onset] = density_by_position.get(onset, 0) + 1
    return density_by_position


def get_filling_submelody_contour(filling_notes: 'NoteSeq') -> dict:
    """
    Calculate the filling sub-melody contour: for each onset position,
    find the highest note (by pitch) among filling notes.

    Args:
        filling_notes: NoteSeq object containing only filling notes (each note must have .onset and .pitch).

    Returns:
        Dictionary mapping onset position (int) to the highest filling note (Note object).
    """
    contour = {}
    onset_dict = {}
    for note in filling_notes.notes:
        onset = note.onset // 6  # Convert to 8th note position if needed
        onset_dict.setdefault(onset, []).append(note)
    for onset, notes_at_onset in onset_dict.items():
        highest_note = max(notes_at_onset, key=lambda n: n.pitch)
        contour[onset] = highest_note
    return contour


if __name__ == '__main__':
    main()