from remi_z import MultiTrack, Bar
class Guitar():
    def __init__(self,tuning=[40, 45, 50, 55, 59, 64], fretnum=24):
        self.tuning= tuning
        self.fretnum=fretnum
        def genfboard():
            boarddic={}
            stri=6
            for i in self.tuning:
                boarddic[stri]=[]
                for j in range(self.fretnum+1):
                    boarddic[stri].append(i+j)
                stri-=1
            return boarddic
        self.fboard=genfboard()
        self.cagedshape={
            "C": {6: 0, 5: 3, 4: 2, 3: 0, 2: 1, 1: 0},
            "A": {6: 0, 5: 0, 4: 2, 3: 2, 2: 2, 1: 0},
            "G": {6: 3, 5: 2, 4: 0, 3: 0, 2: 0, 1: 3},
            "E": {6: 0, 5: 2, 4: 2, 3: 1, 2: 0, 1: 0},
            "D": {6: 0, 5: 0, 4: 0, 3: 2, 2: 3, 1: 2}# (Dshape,6弦应该mute掉)
        }
        self.chords = {
            #C major& A minor
            "C": [  # 再细化，只包含可替换的和弦，根据chords with color来
                    # CAGED 根据规则
                {6: 0, 5: 3, 4: 2, 3: 0, 2: 1, 1: 0},#Triad
                {6: 0, 5: 3, 4: 0, 3: 0, 2: 3, 1: 3},#sus2
                {6: 0, 5: 3, 4: 3, 3: 0, 2: 1, 1: 3},#sus4
                {6: 0, 5: 3, 4: 2, 3: 0, 2: 3, 1: 3},#add9
                {6: 0, 5: 3, 4: 2, 3: 2, 2: 1, 1: 0},#6
                {6: 0, 5: 3, 4: 2, 3: 0, 2: 0, 1: 0},#7
                {6: 0, 5: 3, 4: 2, 3: 4, 2: 3, 1: 0}#9
            ],
            "D": [
                {6: 0, 5: 0, 4: 0, 3: 2, 2: 3, 1: 1},#Triad
                {6: 0, 5: 0, 4: 0, 3: 2, 2: 3, 1: 0},#sus2
                {6: 0, 5: 0, 4: 0, 3: 2, 2: 3, 1: 3},#sus4
                {6: 0, 5: 0, 4: 3, 3: 2, 2: 3, 1: 0},#add9
                {6: 0, 5: 0, 4: 0, 3: 2, 2: 0, 1: 1},#6
                {6: 0, 5: 0, 4: 0, 3: 2, 2: 1, 1: 1},#7
                {6: 0, 5: 5, 4: 3, 3: 5, 2: 5, 1: 0}#9
            ],
            "E": [
                {6: 0, 5: 2, 4: 2, 3: 0, 2: 0, 1: 0},#Triad
                {6: 0, 5: 2, 4: 2, 3: 2, 2: 0, 1: 0},#sus4
                {6: 0, 5: 2, 4: 2, 3: 0, 2: 3, 1: 0}#7
            ],
            "F": [
                {6: 1, 5: 3, 4: 3, 3: 2, 2: 1, 1: 1},#Triad
                {6: 0, 5: 3, 4: 3, 3: 0, 2: 1, 1: 1},#sus2
                {6: 0, 5: 0, 4: 3, 3: 2, 2: 1, 1: 3},#add9
                {6: 0, 5: 0, 4: 3, 3: 2, 2: 3, 1: 1},#6
                {6: 0, 5: 0, 4: 3, 3: 2, 2: 1, 1: 0},#7
                {6: 0, 5: 8, 4: 7, 3: 9, 2: 8, 1: 0}#9

            ],
            "G": [
                {6: 3, 5: 2, 4: 0, 3: 0, 2: 0, 1: 3},#Triad
                {6: 3, 5: 0, 4: 0, 3: 2, 2: 3, 1: 3},#sus2
                {6: 3, 5: 3, 4: 0, 3: 0, 2: 1, 1: 3},#sus4
                {6: 3, 5: 0, 4: 0, 3: 2, 2: 0, 1: 3},#add9
                {6: 3, 5: 2, 4: 0, 3: 0, 2: 3, 1: 0},#6 
                {6: 3, 5: 2, 4: 3, 3: 0, 2: 0, 1: 3}#7
            ],
            "A": [
                {6: 0, 5: 0, 4: 2, 3: 2, 2: 1, 1: 0},#Triad
                {6: 0, 5: 0, 4: 2, 3: 2, 2: 0, 1: 0},#sus2
                {6: 0, 5: 0, 4: 2, 3: 2, 2: 3, 1: 0},#sus4
                {6: 0, 5: 0, 4: 2, 3: 4, 2: 1, 1: 0},#add9
                {6: 0, 5: 0, 4: 2, 3: 0, 2: 1, 1: 0},#7

            ],
            "B": [
                {6: 0, 5: 2, 4: 3, 3: 4, 2: 3, 1: 0},#Triad
                {6: 0, 5: 2, 4: 3, 3: 2, 2: 3, 1: 0}#7
            ],
        }

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
        self.chord_rule = {
                        'Major': [0, 4, 7],
                        'Minor': [0, 3, 7],
                        'Augmented': [0, 4, 8],
                        'Diminished': [0, 3, 6],
                        'Major7': [0, 4, 7, 11],      # Major 7th: root, major third, perfect fifth, major seventh
                        'Minor7': [0, 3, 7, 10],      # Minor 7th: root, minor third, perfect fifth, minor seventh
                        'Dominant7': [0, 4, 7, 10],   # Dominant 7th: root, major third, perfect fifth, minor seventh
                        'Sus4': [0, 5, 7],            # Suspended 4th: root, perfect fourth, perfect fifth
                        'Sus2': [0, 2, 7]             # Suspended 2nd: root, major second, perfect fifth
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
            res.append('#')
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

def name2pitch(nameseq):
    res=[]
    for term in nameseq:
        if len(term)==2:
            name=term[0]
            lev=int(term[1])
        else:
            name=term[0:2]
            lev=int(term[2])

        if name=='C':
            name=0
        elif name=='C#':
            name=1
        elif name=='D':
            name=2
        elif name=='D#':
            name=3
        elif name=='E':
            name=4
        elif name=='F':
            name=5
        elif name=='F#':
            name=6
        elif name=='G':
            name=7
        elif name=='G#':
            name=8
        elif name=='A':
            name=9
        elif name=='A#':
            name=10
        elif name=='B':
            name=11
        res.append(name+12*(lev+1))
    return res

def visualize_guitar_tab(sequence):
    """
    Visualizes guitar tablature from a sequence, showing only positions where at least one string is played.
    Displays a time axis (position indices) at the top for played positions.
    Args:
        sequence: List of tuples or lists, each representing a chord (fret positions, optionally with finger assignments).
    """
    # Prepare data: collect only played positions
    played_indices = []
    played_frets = []
    played_fingers = []
    for i, entry in enumerate(sequence):
        if isinstance(entry, tuple):
            frets, fingers = entry
        else:
            frets = entry
            fingers = None
        if any(fret != -1 for fret in frets):
            played_indices.append(i)
            played_frets.append(frets)
            played_fingers.append(fingers)

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

    # Print time axis
    print('Pos:  ', end='')
    for idx in played_indices:
        print(str(idx).rjust(2), end='-')
    print()

    # Print finger positions if available
    if any(f is not None for f in played_fingers):
        print("Finger:")
        for line in finger_lines:
            print(line)
        print()

    print("Tab:")
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

def tablature_to_multitrack(tablature, guitar, bar_id=0, time_signature=(4,4), tempo=120, velocity=100, duration=12, inst_id=25):
    """
    Convert a single-bar tablature (list of chords) to a MultiTrack object.
    Args:
        tablature (list): List of chords, each chord is a list of fret numbers.
        guitar (Guitar): Your Guitar class instance.
        bar_id (int): Bar index.
        time_signature (tuple): Time signature for the bar.
        tempo (float): Tempo for the bar.
        velocity (int): MIDI velocity for all notes.
        duration (int): Duration (in REMI-z ticks, 12 = 16th note).
        inst_id (int): MIDI program number (25 = Acoustic Guitar).
    Returns:
        MultiTrack: A MultiTrack object representing the tablature.
    """
    notes_of_insts = {inst_id: {}}  # {inst_id: {onset: [[pitch, duration, velocity], ...]}}
    for onset, chord in enumerate(tablature):
        midi_notes = guitar.get_chord_midi({i+1: fret for i, fret in enumerate(chord)})
        for note in midi_notes:
            if note > 0:
                if onset * duration not in notes_of_insts[inst_id]:
                    notes_of_insts[inst_id][onset * duration] = []
                notes_of_insts[inst_id][onset * duration].append([note, duration, velocity])
    bar = Bar(id=bar_id, notes_of_insts=notes_of_insts, time_signature=time_signature, tempo=tempo)
    mt = MultiTrack([bar])
    return mt

def tablature_to_midi(tablature, guitar, output_path, **kwargs):
    """
    Convert a tablature to a MIDI file using REMI-z's MultiTrack and Bar.
    Args:
        tablature (list): List of chords, each chord is a list of fret numbers.
        guitar (Guitar): Your Guitar class instance.
        output_path (str): Path to save the MIDI file.
        kwargs: Additional arguments for tablature_to_multitrack.
    Usage:
        tablature_to_midi(best_tablature, guitar, "output.mid")
    """
    mt = tablature_to_multitrack(tablature, guitar, **kwargs)
    mt.to_midi(output_path)

def multi_bar_tablature_to_midi(tab_bars, guitar, output_path, **kwargs):
    """
    Convert a multi-bar arrangement (list of bars, each a list of chords) to a MIDI file.
    Args:
        tab_bars (list of list): Each element is a bar (list of chords).
        guitar (Guitar): Your Guitar class instance.
        output_path (str): Path to save the MIDI file.
        kwargs: Additional arguments for tablature_to_multitrack (e.g., time_signature, tempo, etc.).
    """
    bars = []
    for bar_id, tab in enumerate(tab_bars):
        bar = tablature_to_multitrack(tab, guitar, bar_id=bar_id, **kwargs).bars[0]
        bars.append(bar)
    mt = MultiTrack(bars)
    mt.to_midi(output_path)