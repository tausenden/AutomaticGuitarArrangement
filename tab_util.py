from fretboard_util import Fretboard
import numpy as np
from remi_z import Bar, MultiTrack
from typing import List

class Chart:
    '''
    This class represent a chord chart on guitar neck.
    '''

    def update_statistics(self):
        """Update self.position (lowest nonzero fret) and self.avg_fret (average nonzero fret)."""
        nonzero_frets = [sf[1] for sf in self.string_fret_list if sf[1] != 0]
        if nonzero_frets:
            self.position = min(nonzero_frets)
            self.avg_fret = sum(nonzero_frets) / len(nonzero_frets)
        else:
            self.position = 0
            self.avg_fret = 0

    def __init__(self,
                 string_fret_list=None,
                 display_note_name=True,
                 chord_name=None,
                 melody_list=None,
                 ):
        self.string_fret_list = string_fret_list if string_fret_list is not None else []
        self.fretboard = Fretboard()
        self.display_note_name = display_note_name
        self.update_statistics()

        self.chord_name = chord_name
        self.melody_list = melody_list

    def __str__(self):
        num_strings = 6
        num_frets = 5
        string_names = ['E', 'A', 'D', 'G', 'B', 'e']  # low to high
        fretboard = self.fretboard
        # string_fret_list: list of (string_id, fret), string_id: 6 (low E, leftmost) to 1 (high e, rightmost)
        # Find all nonzero frets
        fretted = [item[1] for item in self.string_fret_list if item[1] != 0]
        if fretted:
            min_fret = min(fretted)
            max_fret = max(fretted)
        else:
            min_fret = 1
            max_fret = 1
        # Special case: if highest fret <= 5, always show 1-5 (do not use self.position for this)
        if max_fret <= 5:
            min_fret = 1
            max_fret = 5
        # If highest fret > 5 and self.position is set, use it as min_fret
        elif self.position is not None:
            min_fret = self.position
            max_fret = max([min_fret] + fretted)
            # Check all nonzero frets fit in window
            for f in fretted:
                if not (min_fret <= f <= min_fret + num_frets - 1):
                    raise ValueError("Chord note out of chart window (position set)")
        # Check span if auto window
        elif max_fret - min_fret > num_frets - 1:
            raise ValueError("Chord spans more than 5 frets")
        # Prepare grid
        grid = [[' ' for _ in range(num_strings)] for _ in range(num_frets)]
        # For each string, determine if it is open (● or note name), muted (X), or neither
        open_or_muted = [' ' for _ in range(num_strings)]
        for s in range(num_strings):
            has_fret = False
            is_open = False
            for item in self.string_fret_list:
                string_id, fret = item
                grid_idx = 6 - string_id  # Map string_id 6-1 to grid index 0-5
                if grid_idx == s:
                    if fret == 0:
                        is_open = True
                    else:
                        has_fret = True
            if is_open:
                if self.display_note_name:
                    # Map grid index back to string_id for Fretboard
                    string_id = 6 - s
                    open_or_muted[s] = fretboard.get_note_class(string_id, 0)
                else:
                    open_or_muted[s] = '●'  # Use solid dot for open string
            elif not has_fret:
                open_or_muted[s] = 'X'
            else:
                open_or_muted[s] = ' '
        # Place notes (always use 2-tuple, infer note name)
        for item in self.string_fret_list:
            string_id, fret = item
            grid_idx = 6 - string_id  # Map string_id 6-1 to grid index 0-5
            if fret == 0:
                continue  # Already handled above the nut
            elif min_fret <= fret <= min_fret + num_frets - 1:
                if self.display_note_name:
                    note = fretboard.get_note_class(string_id, fret)
                else:
                    note = '●'
                grid[fret - min_fret][grid_idx] = note
            else:
                raise ValueError("Chord note out of chart window")
        # Build chart lines
        nut_line = '   ' + '-' * (num_strings * 3)
        lines = [nut_line]
        # X/O/note name for muted/open strings above the nut
        xo_line = '   '
        for s in range(num_strings):
            xo_line += open_or_muted[s].center(3)
        lines.append(xo_line)
        # Add horizontal line (the nut or start of chart)
        lines.append(nut_line)
        # Fret grid
        for fret in range(num_frets):
            fret_num = str(min_fret + fret).rjust(2)
            line = fret_num + ' '
            for s in range(num_strings):
                cell = grid[fret][s]
                if cell != ' ':
                    line += cell.center(3)
                else:
                    line += ' | '
            lines.append(line)
        # Add horizontal line to the bottom for aesthetics
        lines.append(nut_line)
        return '\n'.join(lines)

    def __repr__(self):
        return f"Chart(position={self.position}, avg_fret={self.avg_fret:.2f})"
    
    def __eq__(self, other):
        return self.position == other.position and self.string_fret_list == other.string_fret_list
    
    def get_position(self):
        return self.position

    def get_avg_fret(self):
        '''
        Seems the "position" is not a good indicator,
        because there might be multiple executable positions for this chart.
        So here we calculate the average fret of the chart.
        '''
        pressed_frets = []
        for sf in self.string_fret_list:
            if sf[1] != 0:
                pressed_frets.append(sf[1])
        if len(pressed_frets) > 0:
            avg_fret = sum(pressed_frets) / len(pressed_frets)
        else:
            avg_fret = 0
        
        return avg_fret
    
    def fret_more_note(self, notes_sfs, melody_list=None, chord_name=None):
        """Extend the string_fret_list by adding a list of (string_id, fret) tuples."""
        self.string_fret_list.extend(notes_sfs)
        self.update_statistics()

        if melody_list:
            self.melody_list = melody_list
        if chord_name:
            self.chord_name = chord_name

    def get_sf_from_note_name(self, note_name):
        """
        Get the string_fret entry for a given note name.
        Returns (string_id, fret) tuple or None if not found.
        """
        for sf in self.string_fret_list:
            string_id, fret = sf
            if self.fretboard.get_note_name(string_id, fret) == note_name:
                return sf
        return None
    

class Tab:
    """
    This class represents a guitar tab.
    It contains a list of Chart objects.
    """

    def __init__(self, n_positions=8):
        '''
        Initialize an empty tab.

        n_positions: Number of positions in the tab, default is 8.
        '''
        n_strings = 6
        self.matrix = np.full((n_strings, n_positions), -1, dtype=int)
        self.chord_dict = {}  # Maps position_id to chord name

    def __str__(self):
        """
        Return a string representation of the tab.
        Each row represents a string, each column a position.
        -1 means no note pressed.
        """
        lines = []
        for s in range(self.matrix.shape[0]):
            line = ' '.join(['-' if self.matrix[s, p] == -1 else str(self.matrix[s, p]) for p in range(self.matrix.shape[1])])
            lines.append(line)
        return '\n'.join(lines)

    def add_note(self, position_id, string_id, fret):
        '''
        Add a note to the tab at the specified string and position.
        String_id: 1~6 (1 is high e, 6 is low E).
        Position: 0~n_positions-1 (0 is the first position).
        '''
        if string_id < 1 or string_id > 6:
            raise ValueError("String ID must be between 1 and 6.")
        if position_id < 0 or position_id >= self.matrix.shape[1]:
            raise ValueError(f"Position must be between 0 and {self.matrix.shape[1] - 1}.")
        self.matrix[string_id - 1, position_id] = fret

    def add_chord(self, position_id, chord_name:str):
        '''
        Add a chord to the tab at the specified position.
        Chord is a Chart object.
        '''
        if position_id < 0 or position_id >= self.matrix.shape[1]:
            raise ValueError(f"Position must be between 0 and {self.matrix.shape[1] - 1}.")
        self.chord_dict[position_id] = chord_name 

    def convert_to_bar(self):
        """
        Convert the Tab to a Bar object.
        Each pressed fret in self.matrix is converted to a note.
        onset and duration are both multiplied by 6.
        """
        notes = {}
        n_positions = self.matrix.shape[1]
        n_strings = self.matrix.shape[0]
        for p_pos in range(n_positions):
            for s in range(n_strings):
                fret = self.matrix[s, p_pos]
                if fret != -1:
                    # Calculate pitch using standard tuning (EADGBE)
                    # string_id: 1 (high e) to 6 (low E), s: 0 (low E) to 5 (high e)
                    string_id = s + 1
                    fretboard = Fretboard()
                    pitch = fretboard.get_pitch(s, fret)
                    onset = p_pos * 6
                    dur = 6  # Default duration, you can adjust if needed
                    velocity = 96  # Default velocity
                    note = [int(pitch), dur, velocity]
                    if onset not in notes:
                        notes[onset] = []
                    notes[onset].append(note)
        time_signature = (4, 4) # Default time signature, can be adjusted as needed
        tempo = 120  # Default tempo, can be adjusted as needed
        return Bar(id=-1, notes_of_insts={0:notes}, time_signature=time_signature, tempo=tempo)

class TabSeq:
    '''
    A container for multiple Tabs.
    For user-friendly display of song-level tabs.
    '''
    def __init__(self, tab_list:List[Tab]=None):
        '''
        Initialize with a list of Tab objects.
        '''
        self.tab_list = tab_list if tab_list is not None else []

    def __str__(self):
        """
        Return a string representation of the TabSeq.
        Each line shows 4 Tabs in a row, properly aligned.
        The first row displays chord names if available, using Tab.chord_dict.
        The first chord is aligned with the beginning of the bar,
        and the second chord is aligned right after the middle of the bar.
        """
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
                        chord_line_pos = first_pos * 2  # Each position is 2 chars (including space)
                        chord_line[chord_line_pos:chord_line_pos+len(chord1)] = chord1
                    if len(sorted_positions) > 1:
                        second_pos = sorted_positions[1]
                        chord2 = tab.chord_dict[second_pos]
                        chord_line_pos = second_pos * 2
                        chord_line[chord_line_pos:chord_line_pos+len(chord2)] = chord2
                chord_name_lines.append(''.join(chord_line))
            lines.append('   '.join(chord_name_lines))
            # For each line in the tab, join horizontally
            for line_idx in range(tab_height):
                row_line = '   '.join(tab[line_idx] for tab in row_tabs)
                lines.append(row_line)
        return '\n'.join(lines)
    
    def save_to_file(self, filename):
        """
        Save the TabSeq to a text file.
        """
        with open(filename, 'w') as f:
            f.write(str(self))

    def convert_to_note_seq(self):
        """
        Convert the TabSeq to a NoteSeq object.
        This is a placeholder method, as NoteSeq is not defined in this context.
        """
        # Placeholder for conversion logic
        # Assuming NoteSeq is defined elsewhere
        '''
        Convert the Bar object to a piano roll matrix.
        '''
        notes = {}
        bars = []
        for tab in self.tab_list:
            bar = tab.convert_to_bar()
            bars.append(bar)
        mt = MultiTrack.from_bars(bars)
        return mt