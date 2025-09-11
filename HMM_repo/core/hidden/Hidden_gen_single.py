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
        
        # Number of fingers used
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
        
        return forms
    
    def generate_and_save_forms(self, output_file='guitar_forms_single.json'):
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
        
        with open(output_file, 'w') as f:
            json.dump(unique_forms, f, indent=2)
        
        return unique_forms
if __name__ == "__main__":
    generator = FormsGenerator()
    generator.generate_and_save_forms()