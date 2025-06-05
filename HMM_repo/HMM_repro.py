import numpy as np
import json
class HMMrepro:
    """
    Guitar HMM following the paper exactly without improvements
    """
    
    def __init__(self, single_forms_file='guitar_forms_single.json', multi_forms_file='guitar_forms_multi.json'):
        # Guitar configuration
        self.num_strings = 6
        self.num_frets = 19
        self.open_strings = [40, 45, 50, 55, 59, 64]
        self.string_names = ['E', 'A', 'D', 'G', 'B', 'E']
        
        # Load forms from both files
        single_forms = self._load_forms(single_forms_file)
        multi_forms = self._load_forms(multi_forms_file)
        self.forms = single_forms + multi_forms
        print(f"Loaded {len(single_forms)} single forms from {single_forms_file}")
        print(f"Loaded {len(multi_forms)} multi forms from {multi_forms_file}")
        print(f"Total {len(self.forms)} forms")
        
        # Precompute form groups by pitch for efficiency
        self._group_forms_by_pitch()
    
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
        Viterbi algorithm implementation
        """
        T = len(pitch_sequence)
        N = len(self.forms)
        
        if time_intervals is None:
            time_intervals = [1.0] * (T - 1)
        
        # Find valid forms for each pitch
        valid_forms = []
        for pitch in pitch_sequence:
            # Input is always a list, take first element
            pitch = pitch[0] if pitch else -1
            
            if pitch in self.forms_by_pitch:
                valid_forms.append(self.forms_by_pitch[pitch])
            else:
                print(f"No valid forms for pitch {pitch}")
                return None
        
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
        for i in range(6):
            tab_strings.append(f"{self.string_names[5-i]}|")
        
        for i, form in enumerate(path):
            if i > 0 and i % 8 == 0:
                for string_idx in range(6):
                    tab_strings[string_idx] += "|"
            
            for string_idx in range(6):
                display_string = 5 - string_idx
                
                # Handle fret_config - try both int and string keys
                fret_config = form['fret_config']
                if display_string in fret_config:
                    fret = fret_config[display_string]
                elif str(display_string) in fret_config:
                    fret = fret_config[str(display_string)]
                else:
                    fret = -1
                
                if fret == -1:
                    tab_strings[string_idx] += "--"
                else:
                    tab_strings[string_idx] += f"{fret:>2}"
                
                tab_strings[string_idx] += "-"
        
        for tab_line in tab_strings:
            print(tab_line)

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

# Test the exact paper reproduction
if __name__ == "__main__":
    print("Testing Guitar HMM - Exact Paper Reproduction")
    print("=" * 60)
    
    hmm = HMMrepro()
    
    # Test 1: List format input
    print("\nTEST 1: C4-C5 (C major scale)")
    melody = [[60], [62], [64], [65], [67], [69], [71], [72]]  # C D E F G A B C
    path = hmm.viterbi(melody)
    if path:
        hmm.visualize_tablature(path)
    
    # Test 2: Extended scale C4-D5
    print("\n\nTEST 2: C4-D5 (Extended C major scale)")
    melody = [[60], [62], [64], [65], [67], [69], [71], [72], [74]]  # C D E F G A B C D
    path = hmm.viterbi(melody)
    if path:
        hmm.visualize_tablature(path)
    
    print("\n\nTEST 3: Caihong 4 bars MIDI file ")
    midi_path= './caihong-4bar.midi'
    melody = get_all_notes_from_midi(midi_path)
    print(f"Extracted melody: {melody}")
    path = hmm.viterbi(melody)
    #print(f"Path found: {path}")
    if path:
        hmm.visualize_tablature(path)

