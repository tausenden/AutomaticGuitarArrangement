import numpy as np
import json
import os
from remi_z import MultiTrack, Bar
from hmm_export import export_hmm_path
import fluidsynth
import fluidsynth

class HMMrepro:
    """
    Guitar HMM following the paper exactly without improvements
    """
    
    def __init__(self, forms_files=['states/guitar_forms_single_expanded.json', 'states/guitar_forms_multi.json']):
        # Guitar configuration
        self.num_strings = 6
        self.num_frets = 20
        self.num_frets = 20
        self.open_strings = [40, 45, 50, 55, 59, 64]
        self.string_names = ['E', 'A', 'D', 'G', 'B', 'E']  # low to high E to match open_strings order
        
        # Load forms from both files
        self.forms = []
        for name in forms_files:
            forms = self._load_forms(name)
            self.forms.extend(forms)
        print(f"Loaded {len(self.forms)} forms from {forms_files}")
        self.forms = []
        for name in forms_files:
            forms = self._load_forms(name)
            self.forms.extend(forms)
        print(f"Loaded {len(self.forms)} forms from {forms_files}")
        
        # Precompute form groups by pitch for efficiency
        self._group_forms_by_pitch()
        
        # Determine available pitch range for octave transposition
        self.min_available_pitch = min(self.forms_by_pitch.keys())
        self.max_available_pitch = max(self.forms_by_pitch.keys())
        print(f"Available pitch range: {self.min_available_pitch} - {self.max_available_pitch}")
    
    def _load_forms(self, forms_file):
        """Load forms from JSON file"""
        with open(forms_file, 'r') as f:
            forms = json.load(f)
        return forms
    
    def _group_forms_by_pitch(self):
        """Group forms by the pitches they can produce"""
        self.forms_by_pitch = {}
        for i, form in enumerate(self.forms):
            for pitch in form['pitches']:
                if pitch not in self.forms_by_pitch:
                    self.forms_by_pitch[pitch] = []
                self.forms_by_pitch[pitch].append(i)
    
    def _transpose_pitch_to_valid_range(self, pitch):
        """
        Transpose a pitch to the valid range by moving it up or down by octaves.
        Returns the transposed pitch and the number of octaves moved.
        """
        if pitch in self.forms_by_pitch:
            return pitch, 0
        
        original_pitch = pitch
        octaves_moved = 0
        
        # Try moving down by octaves if pitch is too high
        while pitch > self.max_available_pitch:
            pitch -= 12
            octaves_moved -= 1
            if pitch in self.forms_by_pitch:
                print(f"Transposed pitch {original_pitch} down {-octaves_moved} octave(s) to {pitch}")
                return pitch, octaves_moved
        
        # Reset and try moving up by octaves if pitch is too low
        pitch = original_pitch
        octaves_moved = 0
        while pitch < self.min_available_pitch:
            pitch += 12
            octaves_moved += 1
            if pitch in self.forms_by_pitch:
                print(f"Transposed pitch {original_pitch} up {octaves_moved} octave(s) to {pitch}")
                return pitch, octaves_moved
        
        print(f"Warning: Could not transpose pitch {original_pitch} to valid range")
        return original_pitch, 0
    
    def initial_probability(self, form):
        """
        Uniform initial probability since paper doesn't specify
        """
        return 1.0 / len(self.forms)
    
    def transition_probability(self, from_form, to_form, time_interval=1.0):
        """
        Calculate transition probability exactly following the paper's formula:
        a_ij(d_t) ∝ (1/2d_t)exp(-|I_i - I_j|/d_t) × 1/(1+I_j) × 1/(1+W_j) × 1/(1+N_j)
        """
        # Movement along the neck |I_i - I_j|
        movement = abs(from_form['index_pos'] - to_form['index_pos'])
        
        # Time interval d_t
        dt = max(time_interval, 0.1)  # Avoid division by zero
        
        # Laplace distribution term: (1/2d_t)exp(-|I_i - I_j|/d_t)
        laplace_factor = (1.0 / (2.0 * dt)) * np.exp(-movement*2 / dt)
        
        # Difficulty factors from paper
        index_factor = 1.0 / (1.0 + to_form['index_pos'])    # 1/(1+I_j)
        width_factor = 1.0 / (1.0 + to_form['width'])        # 1/(1+W_j)  
        finger_factor = 1.0 / (1.0 + to_form['fingers'])     # 1/(1+N_j)
        
        # Combined probability as in paper
        prob = laplace_factor * index_factor * width_factor * finger_factor
        
        return prob
    
    def output_probability(self, form, target_pitch):
        """
        Output probability - deterministic for single notes as in paper
        """
        # Input is always a list, take first element
        target_pitch = target_pitch[0] if target_pitch else -1
        
        return 1.0 if target_pitch in form['pitches'] else 0.0
    
    def viterbi(self, pitch_sequence, time_intervals=None):
        """
        Viterbi algorithm implementation with automatic pitch transposition
        """
        T = len(pitch_sequence)
        N = len(self.forms)
        
        if time_intervals is None:
            time_intervals = [1.0] * (T - 1)
        
        # Find valid forms for each pitch, with automatic transposition
        valid_forms = []
        transposed_pitches = []
        for pitch in pitch_sequence:
            # Input is always a list, take first element
            original_pitch = pitch[0] if pitch else -1
            
            if original_pitch in self.forms_by_pitch:
                valid_forms.append(self.forms_by_pitch[original_pitch])
                transposed_pitches.append([original_pitch])
            else:
                # Try to transpose the pitch to a valid range
                transposed_pitch, octaves_moved = self._transpose_pitch_to_valid_range(original_pitch)
                
                if transposed_pitch in self.forms_by_pitch:
                    valid_forms.append(self.forms_by_pitch[transposed_pitch])
                    transposed_pitches.append([transposed_pitch])
                else:
                    print(f"Failed to find valid arrangement even after transposition for pitch {original_pitch}")
                    return None
        
        # Update pitch_sequence to use transposed pitches for the rest of the algorithm
        pitch_sequence = transposed_pitches
        
        # Initialize Viterbi tables
        delta = np.full((T, N), -np.inf)
        psi = np.zeros((T, N), dtype=int)
        
        # Initialize first time step
        for i in valid_forms[0]:
            initial_prob = self.initial_probability(self.forms[i])
            output_prob = self.output_probability(self.forms[i], pitch_sequence[0])
            if initial_prob > 0 and output_prob > 0:
                delta[0, i] = np.log(initial_prob * output_prob)
        
        # Forward pass
        for t in range(1, T):
            for j in valid_forms[t]:
                max_prob = -np.inf
                best_prev = -1
                
                for i in valid_forms[t-1]:
                    if delta[t-1, i] > -np.inf:
                        trans_prob = self.transition_probability(
                            self.forms[i], 
                            self.forms[j], 
                            time_intervals[t-1]
                        )
                        
                        if trans_prob > 0:
                            prob = delta[t-1, i] + np.log(trans_prob)
                            
                            if prob > max_prob:
                                max_prob = prob
                                best_prev = i
                
                if max_prob > -np.inf:
                    output_prob = self.output_probability(self.forms[j], pitch_sequence[t])
                    if output_prob > 0:
                        delta[t, j] = max_prob + np.log(output_prob)
                        psi[t, j] = best_prev
        
        # Find best final state
        final_valid = [i for i in valid_forms[-1] if delta[T-1, i] > -np.inf]
        if not final_valid:
            print("No valid path found")
            return None
        
        best_final = max(final_valid, key=lambda i: delta[T-1, i])
        
        # Backtrack
        path_indices = [0] * T
        path_indices[T-1] = best_final
        
        for t in range(T-2, -1, -1):
            path_indices[t] = psi[t+1, path_indices[t+1]]
        
        # Convert to forms
        path = [self.forms[i] for i in path_indices]
        
        # Print debug info
        positions = [form['index_pos'] for form in path]
        print(f"Index positions: {positions}")
        
        return path
    
    def visualize_tablature(self, path):
        """Create tablature visualization"""
        tab_strings = []
        # Display strings from high to low (standard tab format)
        # self.string_names = ['E', 'A', 'D', 'G', 'B', 'E'] (low to high)
        # Display order should be: E(high), B, G, D, A, E(low)
        for i in range(6):
            tab_strings.append(f"{self.string_names[5-i]}|")
        
        for i, form in enumerate(path):
            if i > 0 and i % 8 == 0:
                for string_idx in range(6):
                    tab_strings[string_idx] += "|"
            
            for string_idx in range(6):
                # string_idx 0 = high E display line, should get data from fret_config["5"] (high E)
                # string_idx 1 = B display line, should get data from fret_config["4"] (B string)
                # string_idx 2 = G display line, should get data from fret_config["3"] (G string)
                # string_idx 3 = D display line, should get data from fret_config["2"] (D string)
                # string_idx 4 = A display line, should get data from fret_config["1"] (A string)
                # string_idx 5 = low E display line, should get data from fret_config["0"] (low E)
                fret_config_key = str(5 - string_idx)
                
                # Handle fret_config - keys are strings "0" through "5"
                fret_config = form['fret_config']
                if fret_config_key in fret_config:
                    fret = fret_config[fret_config_key]
                else:
                    fret = -1
                
                if fret == -1:
                    tab_strings[string_idx] += "--"
                else:
                    tab_strings[string_idx] += f"{fret:>2}"
                
                tab_strings[string_idx] += "-"
        
        for tab_line in tab_strings:
            print(tab_line)

    def path_to_midi(self, path, output_path, tempo=120, time_signature=(4, 4), velocity=96, duration=6):
        """
        Convert HMM path to MIDI file using REMI-z.
        
        Args:
            path: List of form dictionaries from HMM viterbi algorithm
            output_path: Path to save the MIDI file
            tempo: Tempo in BPM (default: 120)
            time_signature: Time signature as tuple (numerator, denominator) (default: (4, 4))
            velocity: MIDI velocity for all notes (default: 96)
            duration: Duration in REMI-z ticks (default: 6 = 16th note)
        """
        if not path:
            print("Empty path provided")
            return None
        
        # Create notes dictionary for the bar
        notes = {}
        
        # Process each form in the path
        for time_step, form in enumerate(path):
            onset_time = time_step
            
            # Get fret configuration
            fret_config = form['fret_config']
            
            # Process each string
            for string_idx in range(6):
                # Get fret for this string (handle both int and string keys)
                if string_idx in fret_config:
                    fret = fret_config[string_idx]
                elif str(string_idx) in fret_config:
                    fret = fret_config[str(string_idx)]
                else:
                    fret = -1
                
                # If fret is valid (not -1), create a note
                if fret >= 0:
                    # Calculate MIDI pitch based on string and fret
                    # String indices: 0=low E, 1=A, 2=D, 3=G, 4=B, 5=high E
                    # Open string pitches: [40, 45, 50, 55, 59, 64]
                    open_pitch = self.open_strings[string_idx]
                    midi_pitch = open_pitch + fret
                    
                    # Add note to the onset time
                    if onset_time not in notes:
                        notes[onset_time] = []
                    
                    note_data = [midi_pitch, duration, velocity]
                    notes[onset_time].append(note_data)
        
        # Create Bar object
        bar = Bar(
            id=0,
            notes_of_insts={0: notes},  # Use instrument 0 (acoustic guitar)
            time_signature=time_signature,
            tempo=tempo
        )
        
        # Create MultiTrack and save to MIDI
        mt = MultiTrack([bar])
        mt.to_midi(output_path)
        
        print(f"MIDI file saved to: {output_path}")
        return mt

    def path_to_multitrack(self, path, tempo=120, time_signature=(4, 4), velocity=96, duration=6):
        """
        Convert HMM path to MultiTrack object without saving to file.
        
        Args:
            path: List of form dictionaries from HMM viterbi algorithm
            tempo: Tempo in BPM (default: 120)
            time_signature: Time signature as tuple (numerator, denominator) (default: (4, 4))
            velocity: MIDI velocity for all notes (default: 96)
            duration: Duration in REMI-z ticks (default: 6 = 16th note)
            
        Returns:
            MultiTrack object
        """
        if not path:
            print("Empty path provided")
            return None
        
        # Create notes dictionary for the bar
        notes = {}
        
        # Process each form in the path
        for time_step, form in enumerate(path):
            onset_time = time_step * duration
            
            # Get fret configuration
            fret_config = form['fret_config']
            
            # Process each string
            for string_idx in range(6):
                # Get fret for this string (handle both int and string keys)
                if string_idx in fret_config:
                    fret = fret_config[string_idx]
                elif str(string_idx) in fret_config:
                    fret = fret_config[str(string_idx)]
                else:
                    fret = -1
                
                # If fret is valid (not -1), create a note
                if fret >= 0:
                    # Calculate MIDI pitch based on string and fret
                    open_pitch = self.open_strings[string_idx]
                    midi_pitch = open_pitch + fret
                    
                    # Add note to the onset time
                    if onset_time not in notes:
                        notes[onset_time] = []
                    
                    note_data = [midi_pitch, duration, velocity]
                    notes[onset_time].append(note_data)
        
        # Create Bar object
        bar = Bar(
            id=0,
            notes_of_insts={0: notes},
            time_signature=time_signature,
            tempo=tempo
        )
        
        # Create and return MultiTrack
        return MultiTrack([bar])

def get_all_notes_from_midi(midi_file_path):
    """
    Extract all notes from a MIDI file as a list of lists with pitch values.
    """
    from remi_z import MultiTrack
    
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
            position_notes[pos] = list(set(position_notes[pos]))
        
        sorted_positions = sorted(position_notes.keys())
        for pos in sorted_positions:
            all_positions.append(position_notes[pos])
    
    return all_positions

def hmm_path_to_midi(path, output_path, open_strings=None, tempo=120, time_signature=(4, 4), velocity=96, duration=6):
    """
    Standalone function to convert HMM path to MIDI file.
    
    Args:
        path: List of form dictionaries from HMM viterbi algorithm
        output_path: Path to save the MIDI file
        open_strings: List of open string MIDI pitches [low_E, A, D, G, B, high_E] (default: [40, 45, 50, 55, 59, 64])
        tempo: Tempo in BPM (default: 120)
        time_signature: Time signature as tuple (numerator, denominator) (default: (4, 4))
        velocity: MIDI velocity for all notes (default: 96)
        duration: Duration in REMI-z ticks (default: 6 = 16th note)
    """
    if open_strings is None:
        open_strings = [40, 45, 50, 55, 59, 64]  # Standard guitar tuning
    
    if not path:
        print("Empty path provided")
        return None
    
    # Create notes dictionary for the bar
    notes = {}
    
    # Process each form in the path
    for time_step, form in enumerate(path):
        onset_time = time_step * duration//2
        
        # Get fret configuration
        fret_config = form['fret_config']
        
        # Process each string
        for string_idx in range(6):
            # Get fret for this string (handle both int and string keys)
            if string_idx in fret_config:
                fret = fret_config[string_idx]
            elif str(string_idx) in fret_config:
                fret = fret_config[str(string_idx)]
            else:
                fret = -1
            
            # If fret is valid (not -1), create a note
            if fret >= 0:
                # Calculate MIDI pitch based on string and fret
                open_pitch = open_strings[string_idx]
                midi_pitch = open_pitch + fret
                
                # Add note to the onset time
                if onset_time not in notes:
                    notes[onset_time] = []
                
                note_data = [midi_pitch, duration, velocity]
                notes[onset_time].append(note_data)
    
    # Create Bar object
    bar = Bar(
        id=0,
        notes_of_insts={0: notes},  # Use instrument 0 (acoustic guitar)
        time_signature=time_signature,
        tempo=tempo
    )
    
    # Create MultiTrack and save to MIDI
    mt = MultiTrack([bar])
    mt.to_midi(output_path)
    
    print(f"MIDI file saved to: {output_path}")
    return mt

def hmm_path_to_multitrack(path, open_strings=None, tempo=120, time_signature=(4, 4), velocity=96, duration=6):
    """
    Standalone function to convert HMM path to MultiTrack object.
    
    Args:
        path: List of form dictionaries from HMM viterbi algorithm
        open_strings: List of open string MIDI pitches [low_E, A, D, G, B, high_E] (default: [40, 45, 50, 55, 59, 64])
        tempo: Tempo in BPM (default: 120)
        time_signature: Time signature as tuple (numerator, denominator) (default: (4, 4))
        velocity: MIDI velocity for all notes (default: 96)
        duration: Duration in REMI-z ticks (default: 6 = 16th note)
        
    Returns:
        MultiTrack object
    """
    if open_strings is None:
        open_strings = [40, 45, 50, 55, 59, 64]  # Standard guitar tuning
    
    if not path:
        print("Empty path provided")
        return None
    
    # Create notes dictionary for the bar
    notes = {}
    
    # Process each form in the path
    for time_step, form in enumerate(path):
        onset_time = time_step * duration
        
        # Get fret configuration
        fret_config = form['fret_config']
        
        # Process each string
        for string_idx in range(6):
            # Get fret for this string (handle both int and string keys)
            if string_idx in fret_config:
                fret = fret_config[string_idx]
            elif str(string_idx) in fret_config:
                fret = fret_config[str(string_idx)]
            else:
                fret = -1
            
            # If fret is valid (not -1), create a note
            if fret >= 0:
                # Calculate MIDI pitch based on string and fret
                open_pitch = open_strings[string_idx]
                midi_pitch = open_pitch + fret
                
                # Add note to the onset time
                if onset_time not in notes:
                    notes[onset_time] = []
                
                note_data = [midi_pitch, duration, velocity]
                notes[onset_time].append(note_data)
    
    # Create Bar object
    bar = Bar(
        id=0,
        notes_of_insts={0: notes},
        time_signature=time_signature,
        tempo=tempo
    )
    
    # Create and return MultiTrack
    return MultiTrack([bar])

# Test the exact paper reproduction
if __name__ == "__main__":
    print("Testing Guitar HMM - Exact Paper Reproduction")
    print("=" * 60)
    
    hmm = HMMrepro()
    
    # Test 1: List format input
    # print("\nTEST 1: C4-C5 (C major scale)")
    # melody = [[60], [62], [64], [65], [67], [69], [71], [72]]  # C D E F G A B C
    # path = hmm.viterbi(melody)
    # if path:
    #     hmm.visualize_tablature(path)
    
    # # Test 2: Extended scale C4-D5
    # print("\n\nTEST 2: C4-D5 (Extended C major scale)")
    # melody = [[60], [62], [64], [65], [67], [69], [71], [72], [74]]  # C D E F G A B C D
    # path = hmm.viterbi(melody)
    # if path:
    #     hmm.visualize_tablature(path)
    
    print("\n\nTEST 3: Processing all MIDI files in folder")
    
    # Configuration
    input_folder = '../test_arr_midis'  # Fixed input folder
    output_folder = './test_arranged_midis'  # Fixed output folder
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all MIDI files
    midi_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mid', '.midi'))]
    
    for filename in midi_files:
        midi_file_path = os.path.join(input_folder, filename)
        
        # Get output filename (keep same name, normalize .midi to .mid)
        if filename.lower().endswith('.midi'):
            output_name = filename[:-5] + '.mid'
        else:
            output_name = filename
        output_midi_path = os.path.join(output_folder, output_name)
        
        print(f"Processing MIDI file: {midi_file_path}")
        print(f"Output will be saved to: {output_midi_path}")

        melody = get_all_notes_from_midi(midi_file_path)
        path = hmm.viterbi(melody)
        
        if path:
            # Use unified export pipeline (tab, midi, wav)
            song_name = os.path.splitext(output_name)[0]
            export_hmm_path(path, song_name=song_name, positions_per_bar=8, tempo=100, output_dir=output_folder)
        else:
            print(f"Failed to find valid arrangement for {filename}")
        
        print(f"\nProcess completed!")
        print(f"Input: {midi_file_path}")
        print(f"Output: {output_midi_path}")

