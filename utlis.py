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

class Hand():
    def __init__(self):
        self.f1=None
        self.f2=None
        self.f3=None
        self.f4=None

def pitch2name(seq):
    res=[]
    for i in seq:
        lev=i//12-1
        name=i%12
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
    Visualizes guitar tablature and finger positions from a sequence of (fret, finger) tuples.
    Each tuple contains two lists: fret positions and finger assignments for each string.
    
    Args:
        sequence: List of tuples, each containing two lists (fret positions and finger assignments)
    """
    # Initialize strings for both finger positions and fret positions
    strings = ['E|', 'B|', 'G|', 'D|', 'A|', 'E|']
    finger_strings = ['E|', 'B|', 'G|', 'D|', 'A|', 'E|']
    
    # Process each position in the sequence
    for i, (frets, fingers) in enumerate(sequence):
        # Add bar line every 4 positions
        if i > 0 and i % 8 == 0:
            for j in range(6):
                strings[j] += '|'
                finger_strings[j] += '|'
        
        # Process each string
        for string_idx in range(6):
            # Handle fret positions
            fret = frets[string_idx]
            if fret == -1:
                strings[string_idx] += '--'
            else:
                strings[string_idx] += str(fret).rjust(2)
            
            # Handle finger positions
            finger = fingers[string_idx]
            if finger == -1:
                finger_strings[string_idx] += '--'
            else:
                finger_strings[string_idx] += str(finger).rjust(2)
                
        # Add separator
        for j in range(6):
            strings[j] += '-'
            finger_strings[j] += '-'
    
    # Print results
    print("Finger positions:")
    for line in finger_strings:
        print(line)
    print("\nFret positions:")
    for line in strings:
        print(line)