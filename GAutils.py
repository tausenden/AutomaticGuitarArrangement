from mido import midifiles
from remi_z import MultiTrack, Bar
import numpy as np
import random

class Guitar:
    """
    Minimal guitar model aligned to tab_util.Tab conventions:
    - string_id 1 is high e (64), ... string_id 6 is low E (40)
    - fboard[string_id][fret] gives MIDI pitch
    """
    def __init__(self, tuning=[64, 59, 55, 50, 45, 40], fretnum=24):
        self.tuning = tuning  # high E to low E
        self.fretnum = fretnum
        def genfboard():
            mapping = {}
            stri = 1
            for open_pitch in self.tuning:
                mapping[stri] = [open_pitch + j for j in range(self.fretnum + 1)]
                stri += 1
            return mapping
        self.fboard = genfboard()

        self.rootnote={
            'C': 0,
            'C#': 1,
            'D': 2,
            'D#': 3,
            'E': 4,
            'F': 5,
            'F#': 6,
            'G': 7,
            'G#': 8,
            'A': 9,
            'A#': 10,
            'B': 11
        }
        chords4NCC = {}
        for root_name, root_value in self.rootnote.items():
            # Major triad: root, major third (+4), perfect fifth (+7)
            chords4NCC[root_name] = [root_value, (root_value + 4) % 12, (root_value + 7) % 12]
        self.chords4NCC = chords4NCC    

    def get_chord_midi(self, chord_fingering):

        midi_notes = []
        for string, fret in chord_fingering.items():
            if fret != -1:
                midi_note = self.fboard[string][fret]
            else:
                midi_note=-1
            midi_notes.append(midi_note)

        return midi_notes

def pitch2name(seq):
    res=[]
    for i in seq:
        lev=i//12-1
        name=i%12
        if i==-1:
            res.append('X')
            continue

        if name==0:
            name='C'
        elif name==1:
            name='C#'
        elif name==2:
            name='D'
        elif name==3:
            name='D#'
        elif name==4:
            name='E'
        elif name==5:
            name='F'
        elif name==6:
            name='F#'
        elif name==7:
            name='G'
        elif name==8:
            name='G#'
        elif name==9:
            name='A'
        elif name==10:
            name='A#'
        elif name==11:
            name='B'
        res.append(name+str(lev))
    return res



class GATab:
    """
    GATab class - functionally identical to tab_util.Tab.
    This class represents a guitar tab for GA results.
    Matrix-based 6xN structure where each position represents a time slot.
    """
    def __init__(self, n_positions=8, bar_data=None, original_onsets=None):
        """
        Initialize a GATab.
        
        Args:
            n_positions: Number of positions in the tab, default is 8 (used when bar_data is None)
            bar_data: List of lists representing chord fingerings (ga_reproduction.py format)
            original_onsets: List of onset times (for compatibility with ga_reproduction.py)
        """
        n_strings = 6
        
        # Handle bar_data format from ga_reproduction.py
        if bar_data is not None:
            n_positions = len(bar_data)
            self.matrix = np.full((n_strings, n_positions), -1, dtype=int)
            # Convert bar_data (list of lists) to matrix format
            for pos, chord in enumerate(bar_data):
                for string_idx, fret in enumerate(chord):
                    if pos < self.matrix.shape[1] and string_idx < self.matrix.shape[0]:
                        self.matrix[string_idx, pos] = fret
        else:
            # Standard Tab-like initialization
            self.matrix = np.full((n_strings, n_positions), -1, dtype=int)
        
        self.chord_dict = {}  # Maps position_id to chord name
        self.original_onsets = original_onsets if original_onsets is not None else []
        self.melody_positions = set()  # For tracking melody positions

    def __str__(self):
        """
        Return a string representation of the tab.
        Each row represents a string, each column a position.
        -1 means no note pressed.
        Each position is 2 characters wide, separated by a single space.
        """
        # Only show positions with notes
        active_positions = [p for p in range(self.matrix.shape[1]) 
                          if any(self.matrix[s, p] != -1 for s in range(self.matrix.shape[0]))]
        lines = []
        for s in range(self.matrix.shape[0]):
            line = ' '.join([
                '--' if self.matrix[s, p] == -1 else str(self.matrix[s, p]).rjust(2)
                for p in active_positions
            ])
            lines.append(line)
        return '\n'.join(lines)

    def add_note(self, position_id, string_id, fret):
        """
        Add a note to the tab at the specified string and position.
        String_id: 1~6 (1 is high e, 6 is low E).
        Position: 0~n_positions-1 (0 is the first position).
        """
        if string_id < 1 or string_id > 6:
            raise ValueError("String ID must be between 1 and 6.")
        if position_id < 0 or position_id >= self.matrix.shape[1]:
            raise ValueError(f"Position must be between 0 and {self.matrix.shape[1] - 1}.")
        self.matrix[string_id - 1, position_id] = fret

    def add_chord(self, position_id, chord_name: str):
        """
        Add a chord to the tab at the specified position.
        """
        if position_id < 0 or position_id >= self.matrix.shape[1]:
            raise ValueError(f"Position must be between 0 and {self.matrix.shape[1] - 1}.")
        self.chord_dict[position_id] = chord_name
    
    def add_chord_info(self, position_id, chord_name: str):
        """
        Add chord information (alias for add_chord for ga_reproduction.py compatibility).
        """
        self.add_chord(position_id, chord_name)
    
    def add_melody_position(self, position_id):
        """
        Mark a position as containing melody notes (for ga_reproduction.py compatibility).
        """
        if position_id < 0 or position_id >= self.matrix.shape[1]:
            raise ValueError(f"Position must be between 0 and {self.matrix.shape[1] - 1}.")
        self.melody_positions.add(position_id)
    
    @property
    def bar_data(self):
        """
        Return the tab as bar_data format (list of lists) for backward compatibility.
        """
        result = []
        for pos in range(self.matrix.shape[1]):
            chord = []
            for string_idx in range(self.matrix.shape[0]):
                chord.append(self.matrix[string_idx, pos])
            result.append(chord)
        return result

    def convert_to_bar(self, guitar):
        """
        Convert the GATab to a Bar object using the provided guitar for pitch mapping.
        Uses original onset values if available, otherwise falls back to position * 6.
        """
        notes = {}
        n_positions = self.matrix.shape[1]
        n_strings = self.matrix.shape[0]
        
        # Find all positions with notes
        active_positions = [p for p in range(n_positions) if any(self.matrix[s, p] != -1 for s in range(n_strings))]
        
        for p_pos in range(n_positions):
            for s in range(n_strings):
                fret = self.matrix[s, p_pos]
                if fret != -1:
                    # Calculate pitch using guitar fretboard
                    string_id = s + 1  # string_id: 1 (high e) to 6 (low E)
                    pitch = guitar.fboard[string_id][fret]
                    
                    # Use original onset if available
                    if self.original_onsets and p_pos in active_positions:
                        active_idx = active_positions.index(p_pos)
                        if active_idx < len(self.original_onsets):
                            onset = self.original_onsets[active_idx]
                        else:
                            onset = p_pos
                    else:
                        onset = p_pos
                    
                    dur = 6  # Default duration, you can adjust if needed
                    velocity = 96  # Default velocity
                    note = [int(pitch), dur, velocity]
                    if onset not in notes:
                        notes[onset] = []
                    notes[onset].append(note)
        time_signature = (4, 4)  # Default time signature, can be adjusted as needed
        tempo = 120  # Default tempo, can be adjusted as needed
        return Bar(id=-1, notes_of_insts={0: notes}, time_signature=time_signature, tempo=tempo)


class GATabSeq:
    """
    GATabSeq class - functionally similar to tab_util.TabSeq.
    A container for multiple GATab objects.
    For user-friendly display of song-level tabs.
    """
    def __init__(self, tab_list=None):
        """
        Initialize with a list of GATab objects.
        """
        self.tab_list = tab_list if tab_list is not None else []
    
    def add_tab(self, tab):
        """Add a GATab to the sequence."""
        self.tab_list.append(tab)
    
    def __str__(self):
        """
        Return a string representation of the GATabSeq.
        Each line shows 4 GATabs in a row, properly aligned.
        The first row displays chord names if available, using GATab.chord_dict.
        """
        if not self.tab_list:
            return "Empty GATabSeq"
        
        tab_lines_list = [str(tab).split('\n') for tab in self.tab_list]
        tab_height = len(tab_lines_list[0]) if tab_lines_list else 0
        lines = []
        
        for i in range(0, len(tab_lines_list), 4):
            row_tabs = tab_lines_list[i:i + 4]
            tab_objs = self.tab_list[i:i + 4]
            
            # Pad with empty tabs if less than 4
            while len(row_tabs) < 4:
                row_tabs.append([' ' * (len(row_tabs[0][0]) if row_tabs else 10)] * tab_height)
                tab_objs.append(None)
            
            # First row: chord names, aligned to positions
            chord_name_lines = []
            position_lines = []
            for tab, tab_lines in zip(tab_objs, row_tabs):
                chord_line = [' ' for _ in range(len(tab_lines[0]))]
                if tab is not None and hasattr(tab, 'chord_dict') and tab.chord_dict:
                    n_positions = len(tab_lines[0].split())
                    # Find positions for first and second chord
                    sorted_positions = sorted(tab.chord_dict.keys())
                    if len(sorted_positions) > 0:
                        first_pos = sorted_positions[0]
                        chord1 = tab.chord_dict[first_pos]
                        # Place chord1 at the start
                        # Each position is 3 chars: 2 for fret, 1 for space
                        chord_line_pos = first_pos * 3
                        chord_line[chord_line_pos:chord_line_pos+len(chord1)] = chord1
                    if len(sorted_positions) > 1:
                        second_pos = sorted_positions[1]
                        chord2 = tab.chord_dict[second_pos]
                        chord_line_pos = second_pos * 3
                        chord_line[chord_line_pos:chord_line_pos+len(chord2)] = chord2
                chord_name_lines.append(''.join(chord_line))
                
                # Position row: show position numbers aligned with tab
                if tab is not None:
                    active_positions = [p for p in range(tab.matrix.shape[1]) 
                                      if any(tab.matrix[s, p] != -1 for s in range(tab.matrix.shape[0]))]
                    position_line = ' '.join([str(p).rjust(2) for p in active_positions])
                else:
                    position_line = ' ' * len(tab_lines[0]) if tab_lines else ''
                position_lines.append(position_line)
            
            lines.append('   '.join(chord_name_lines))
            lines.append('   '.join(position_lines))
            
            # Add segmentation line under position row
            seg_lines = []
            for tab_lines in row_tabs:
                if tab_lines and tab_lines[0]:
                    seg_line = '-' * len(tab_lines[0])
                else:
                    seg_line = ''
                seg_lines.append(seg_line)
            lines.append('   '.join(seg_lines))
            
            # For each line in the tab, join horizontally
            for line_idx in range(tab_height):
                row_line = '   '.join(tab[line_idx] for tab in row_tabs)
                lines.append(row_line)
            lines.append('')
        
        return '\n'.join(lines)
    
    def save_to_file(self, filename):
        """
        Save the GATabSeq to a text file.
        """
        with open(filename, 'w') as f:
            f.write(str(self))
    
    def convert_to_multitrack(self, guitar, tempo=120, time_signature=(4, 4)):
        """
        Convert the GATabSeq to a MultiTrack object.
        Similar to TabSeq.convert_to_note_seq() but returns MultiTrack directly.
        
        Args:
            guitar: Guitar instance for pitch mapping
            tempo: Tempo for the track
            time_signature: Time signature for all bars
            
        Returns:
            MultiTrack object
        """
        from remi_z import MultiTrack
        
        bars = []
        for tab in self.tab_list:
            bar = tab.convert_to_bar(guitar)
            bar.tempo = tempo
            bar.time_signature = time_signature
            bars.append(bar)
        
        return MultiTrack.from_bars(bars)


def visualize_guitar_tab(sequence):
    """
    Visualizes guitar tablature from a sequence, showing only positions where at least one string is played.
    Displays a time axis (position indices or onset times) at the top for played positions.
    Args:
        sequence: Can be one of:
                 - List of lists: [[fret1, fret2, ...], [fret1, fret2, ...], ...] (fret positions only)
                 - Dictionary: {'tab_candi': [...], 'hand_candi': [...]} (from GAimproved)
        show_onset: If True, show onset times (0, 6, 12, 18, ...), if False, show position indices (0, 1, 2, 3, ...)
    """
    # Handle different input formats
    if isinstance(sequence, dict) and 'tab_candi' in sequence:
        # GAimproved format: {'tab_candi': [...], 'hand_candi': [...]}
        tab_candi = sequence['tab_candi']
        hand_candi = sequence.get('hand_candi', None)
        frets_sequence = tab_candi
        fingers_sequence = hand_candi
    elif isinstance(sequence, list):
        # List of lists format: [[frets], [frets], ...] - from ga_reproduction.py
        frets_sequence = sequence
        fingers_sequence = None
    else:
        raise ValueError(f"Unsupported sequence type: {type(sequence)}")
    
    # Prepare data: collect only played positions
    played_indices = []
    played_frets = []
    played_fingers = []
    for i, frets in enumerate(frets_sequence):
        if any(fret != -1 for fret in frets):
            played_indices.append(i)
            played_frets.append(frets)
            if fingers_sequence and i < len(fingers_sequence):
                played_fingers.append(fingers_sequence[i])
            else:
                played_fingers.append(None)

    # Build tab lines
    string_names = ['E', 'B', 'G', 'D', 'A', 'E']
    tab_lines = [name + '|' for name in string_names]
    finger_lines = [name + '|' for name in string_names]

    for frets, fingers in zip(played_frets, played_fingers):
        for string_idx in range(6):
            fret = frets[string_idx]
            if fret == -1:
                tab_lines[string_idx] += '--'
            else:
                tab_lines[string_idx] += str(fret).rjust(2)
            if fingers is not None:
                finger = fingers[string_idx]
                if finger == -1:
                    finger_lines[string_idx] += '--'
                else:
                    finger_lines[string_idx] += str(finger).rjust(2)
        for string_idx in range(6):
            tab_lines[string_idx] += '-'
            if fingers is not None:
                finger_lines[string_idx] += '-'

    # Print time axis - convert position indices to onset times if requested
    # print('os:', end='')
    # for idx in played_indices:
    #     print(str(idx).rjust(2), end='-')
    # print()

    # Print finger positions if available
    if any(f is not None for f in played_fingers):
        print("Finger:")
        print('  ', end='')
        for idx in played_indices:
            print(str(idx).rjust(2), end='-')
        print()
        for line in finger_lines:
            print(line)
        print()

    print("Tab:")
    print('  ', end='')
    for idx in played_indices:
        print(str(idx).rjust(2), end='-')
    print()
    for line in tab_lines:
        print(line)

def midi_process(midi_file_path):
    """
    Process a MIDI file using remi_z for use with Guitar GA.
    This follows the preprocessing logic of GAusing_remiz.py.
    
    Args:
        midi_file_path: Path to the MIDI file
        
    Returns:
        tuple containing:
        - target_melody_list: List of lists of MIDI pitches for each bar
        - target_chord_list: List of chord names (just the root name) for each bar
    """
    
    # Load MIDI file using remi_z
    mt = MultiTrack.from_midi(midi_file_path)
    
    target_melody_list = []
    target_chord_list = []
    
    # Process each bar
    for bar in mt.bars:
        # Extract melody using remi_z's high note method
        mel_notes = bar.get_melody('hi_note')
        
        # Convert to GA format (list of MIDI pitch values)
        melody = []
        for note in mel_notes:
            if note is None:
                melody.append(-1)  # Represent silence as -1
            else:
                melody.append(note.pitch)
        
        target_melody_list.append(melody)
        
        # Extract chord using remi_z's method
        chords = bar.get_chord()
        if chords and chords[0]:
            # Take the first chord of each bar - extract just the root name
            target_chord_list.append(chords[0][0])
        else:
            # Default to C if no chord detected
            target_chord_list.append('C')
    
    return target_melody_list, target_chord_list
def get_all_notes_from_midi(midi_file_path):
    """
    Extract all notes from a MIDI file as a list of lists with pitch values.
    """
    # Load MIDI file using remi_z
    mt = MultiTrack.from_midi(midi_file_path)
    
    # Collect all positions across all bars
    all_positions = []
    
    for bar in mt.bars:
        # Group notes by position within this bar
        position_notes = {}
        all_notes = bar.get_all_notes(include_drum=False)
        
        for note in all_notes:
            pos = note.onset
            if pos not in position_notes:
                position_notes[pos] = []
            position_notes[pos].append(note.pitch)
        
        # Add this bar's positions to the overall list
        sorted_positions = sorted(position_notes.keys())
        for pos in sorted_positions:
            all_positions.append(position_notes[pos])
    
    return all_positions

def set_random(seed=42):
    random.seed(seed)
    np.random.seed(seed)