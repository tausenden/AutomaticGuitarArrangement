from remi_z import MultiTrack
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
            "D": {6: 0, 5: 0, 4: 0, 3: 2, 2: 3, 1: 2}
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

        self.chords4NCC={'C': [0, 4, 7], 'D': [2, 6, 9], 'E': [4, 8, 11], 'F': [5, 9, 0], 'G': [7, 11, 2], 'A': [9, 1, 4], 'B': [11, 3, 6]}
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
        #def genchord(self,chordname):
            

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

def visualize_guitar_tab_withpos(sequence, position_map=None):
    """
    Visualizes guitar tablature and finger positions from a sequence of fret positions,
    including position numbers above the tablature.
    
    Args:
        sequence: List of tuples, each containing either one or two lists 
                 (fret positions and optionally finger assignments)
        position_map: Optional dictionary mapping sequence indices to MIDI positions.
                     If provided, these values will be shown instead of sequence indices.
    """
    # Initialize strings for fret positions
    strings = ['E|', 'B|', 'G|', 'D|', 'A|', 'E|']
    position_row = '  '  # Start with padding to align with string names
    finger_strings = ['E|', 'B|', 'G|', 'D|', 'A|', 'E|']
    
    # Process each position in the sequence
    for i, entry in enumerate(sequence):
        # Check if the entry is a tuple with finger positions
        if isinstance(entry, tuple):
            frets, fingers = entry
        else:
            frets = entry
            fingers = None
        
        # Add bar line every 48 positions
        if i > 0 and i % 48 == 0:
            for j in range(6):
                strings[j] += '|'
                if fingers is not None:
                    finger_strings[j] += '|'
            position_row += ' '
        
        # Check if any string is played at this position
        has_note = any(fret != -1 for fret in frets)
        
        # Determine position number to display
        display_pos = position_map[i] if position_map and i in position_map else i
        
        # Add position number exactly above the tab notation
        if has_note:
            position_row += str(display_pos).rjust(2) + '-'
        else:
            position_row += '---'
        
        # Process each string
        for string_idx in range(6):
            # Handle fret positions
            fret = frets[string_idx]
            if fret == -1:
                strings[string_idx] += '--'
            else:
                strings[string_idx] += str(fret).rjust(2)
            
            # Handle finger positions if available
            if fingers is not None:
                finger = fingers[string_idx]
                if finger == -1:
                    finger_strings[string_idx] += '--'
                else:
                    finger_strings[string_idx] += str(finger).rjust(2)
            
            # Add separator
            strings[string_idx] += '-'
            if fingers is not None:
                finger_strings[string_idx] += '-'
    
    # Print results
    print("Position numbers:")
    print(position_row)
    
    if fingers is not None:
        print("\nFinger positions:")
        for line in finger_strings:
            print(line)
        print()
    
    print("\nFret positions:")
    for line in strings:
        print(line)

def visualize_guitar_tab(sequence):
    """
    Visualizes guitar tablature and finger positions from a sequence of (fret, finger) tuples or just fret positions.
    Each tuple may contain either two lists (fret positions and finger assignments) or just fret positions.
    
    Args:
        sequence: List of tuples, each containing either one or two lists (fret positions and optionally finger assignments)
    """
    # Initialize strings for fret positions
    strings = ['E|', 'B|', 'G|', 'D|', 'A|', 'E|']
    finger_strings = ['E|', 'B|', 'G|', 'D|', 'A|', 'E|']
    
    # Process each position in the sequence
    for i, entry in enumerate(sequence):
        # Check if the entry is a tuple with finger positions
        if isinstance(entry, tuple):
            frets, fingers = entry
        else:
            frets = entry
            fingers = None
        
        # Add bar line every 8 positions
        if i > 0 and i % 8 == 0:
            for j in range(6):
                strings[j] += '|'
                if fingers is not None:
                    finger_strings[j] += '|'
        
        # Process each string
        for string_idx in range(6):
            # Handle fret positions
            fret = frets[string_idx]
            if fret == -1:
                strings[string_idx] += '--'
            else:
                strings[string_idx] += str(fret).rjust(2)
            
            # Handle finger positions if available
            if fingers is not None:
                finger = fingers[string_idx]
                if finger == -1:
                    finger_strings[string_idx] += '--'
                else:
                    finger_strings[string_idx] += str(finger).rjust(2)
                    
        # Add separator
        for j in range(6):
            strings[j] += '-'
            if fingers is not None:
                finger_strings[j] += '-'
    
    # Print results
    # if fingers is not None:
    #     print("Finger positions:")
    #     for line in finger_strings:
    #         print(line)
    #     print()
    
    print("Fret positions:")
    for line in strings:
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