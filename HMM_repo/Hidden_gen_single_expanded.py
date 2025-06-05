import json
import numpy as np

class FormsGenerator:
    def __init__(self):
        self.num_strings = 6
        self.num_frets = 19
        self.open_strings = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, E4
    
    def _create_fret_config(self, fret_assignments):
        """Create fret configuration with string indices 0-5"""
        config = {i: -1 for i in range(self.num_strings)}
        for string, fret in fret_assignments:
            config[string] = fret
        return config
    
    def _calculate_form_properties(self, fret_config, base_position):
        """Calculate index position, width, and number of fingers for a form"""
        # Find all non-open frets
        pressed_frets = [fret for fret in fret_config.values() if fret > 0]
        
        if not pressed_frets:
            # All open strings
            return {
                'index_pos': 0,
                'width': 0,
                'fingers': 0
            }
        
        min_fret = min(pressed_frets)
        max_fret = max(pressed_frets)
        
        # Index position is where the index finger would be
        # Following the paper's approach
        index_pos = base_position
        
        # Width is the span of the form
        width = max_fret - min_fret + 1 if pressed_frets else 0
        
        # Number of fingers used (only count pressed frets, not open strings)
        fingers = len(pressed_frets)
        
        return {
            'index_pos': index_pos,
            'width': width,
            'fingers': fingers
        }
    
    def _generate_single_note_forms_with_positions(self):
        """Generate forms considering hand positions as in the paper"""
        forms = []
        
        for string in range(self.num_strings):
            # Open string form
            pitch = self.open_strings[string]
            if 40 <= pitch <= 83:
                form = {
                    'pitches': [pitch],
                    'fret_config': self._create_fret_config([(string, 0)]),
                    'index_pos': 0,
                    'width': 0,
                    'fingers': 0,
                    'finger_used': [],
                    'string': [string],
                }
                forms.append(form)
            
            # Fretted notes with different hand positions
            for fret in range(1, self.num_frets + 1):
                pitch = self.open_strings[string] + fret
                if 40 <= pitch <= 83:
                    # Generate forms for different hand positions
                    # The index finger can be 0-3 frets behind the pressed fret
                    for finger_used in range(1, 5):  # Which finger presses (1=index, 4=pinky)
                        base_position = max(1, fret - (finger_used - 1))
                        
                        # Skip invalid positions
                        if base_position < 1 or base_position > self.num_frets:
                            continue
                        
                        # 1. Original single note form
                        fret_config = self._create_fret_config([(string, fret)])
                        props = self._calculate_form_properties(fret_config, base_position)
                        
                        form = {
                            'pitches': [pitch],
                            'fret_config': fret_config,
                            'index_pos': props['index_pos'],
                            'width': props['width'],
                            'fingers': props['fingers'],
                            'finger_used': [finger_used],
                            'string': [string],
                        }
                        forms.append(form)
                        
                        # 2. NEW: Add forms with pressed string + each open string
                        for open_string in range(self.num_strings):
                            if open_string == string:
                                continue  # Can't press and play open on same string
                            
                            open_pitch = self.open_strings[open_string]
                            if 40 <= open_pitch <= 83:
                                # Create new form with both pitches
                                combo_fret_config = self._create_fret_config([(string, fret), (open_string, 0)])
                                combo_pitches = sorted([pitch, open_pitch])
                                combo_strings = sorted([string, open_string])
                                
                                combo_form = {
                                    'pitches': combo_pitches,
                                    'fret_config': combo_fret_config,
                                    'index_pos': props['index_pos'],  # Same as single note
                                    'width': props['width'],          # Same as single note
                                    'fingers': props['fingers'],      # Same as single note (only 1 finger used)
                                    'finger_used': [finger_used],     # Same finger
                                    'string': combo_strings,
                                }
                                forms.append(combo_form)
        
        return forms
    
    def generate_and_save_forms(self, output_file='guitar_forms_single_expanded.json'):
        forms = self._generate_single_note_forms_with_positions()
        
        # Remove duplicates based on unique properties
        unique_forms = []
        seen = set()
        
        for form in forms:
            # Create a unique key for each form
            key = (
                tuple(form['pitches']),
                tuple(sorted(form['fret_config'].items())),
                form['index_pos']
            )
            
            if key not in seen:
                seen.add(key)
                unique_forms.append(form)
        
        print(f"Generated {len(unique_forms)} unique forms")
        
        # Show statistics
        single_pitch = len([f for f in unique_forms if len(f['pitches']) == 1])
        two_pitch = len([f for f in unique_forms if len(f['pitches']) == 2])
        print(f"Single pitch forms: {single_pitch}")
        print(f"Two pitch forms: {two_pitch}")
        
        with open(output_file, 'w') as f:
            json.dump(unique_forms, f, indent=2)
        
        return unique_forms

if __name__ == "__main__":
    generator = FormsGenerator()
    generator.generate_and_save_forms()