import json
from itertools import combinations

class MultiFormsGenerator:
    def __init__(self):
        self.open_strings = [40, 45, 50, 55, 59, 64]
    
    def generate_forms_from_basic_form(self, basic_form, max_position=6):
        forms = []
        
        # Find minimum pressed fret
        pressed_frets = [fret for fret in basic_form['fret_config'].values() if fret > 0]
        min_pressed = min(pressed_frets) if pressed_frets else 1
        
        for offset in range(0, max_position - min_pressed + 1):
            # Update fret_config: add offset to pressed strings only
            new_fret_config = {}
            for string, fret in basic_form['fret_config'].items():
                if fret > 0:
                    new_fret_config[string] = fret + offset
                else:
                    new_fret_config[string] = fret
            
            # Calculate all available pitches for this position
            all_pitches = []
            for string, fret in new_fret_config.items():
                if fret >= 0:
                    all_pitches.append(self.open_strings[int(string)] + fret)
            all_pitches = sorted(all_pitches)
            
            # Generate combinations of 2-6 pitches, keeping everything else the same
            for num_pitches in range(2, len(all_pitches)-1):
                for pitch_combo in combinations(all_pitches, num_pitches):
                    new_form = basic_form.copy()
                    
                    # Only change pitches and position-dependent properties
                    new_form['pitches'] = sorted(pitch_combo)
                    new_form['fret_config'] = new_fret_config
                    new_form['index_pos'] = basic_form['index_pos'] + offset
                    
                    # Keep all other properties unchanged (width, fingers, finger_used, string)
                    
                    forms.append(new_form)
        
        return forms
    
    def generate_and_save_forms(self, basic_forms_list, output_file='guitar_forms_multi.json'):
        all_forms = []
        for basic_form in basic_forms_list:
            all_forms.extend(self.generate_forms_from_basic_form(basic_form))
        
        with open(output_file, 'w') as f:
            json.dump(all_forms, f, indent=2)
        
        return all_forms

if __name__ == "__main__":

    generator = MultiFormsGenerator()
    # Your basic form
    basic_form = [
        {
            "pitches": [
            40, 48, 52, 55, 60, 64
            ],
            "fret_config": {
            "0": 0,
            "1": 3,
            "2": 2,
            "3": 0,
            "4": 1,
            "5": 0
            },
            "index_pos": 1,
            "width": 3,
            "fingers": 3,
            "finger_used": [1, 2, 3],
            "string": [1, 2, 4],
        },
        {
            "pitches": [
            40, 45, 50, 57, 62, 66
            ],
            "fret_config": {
            "0": 0,
            "1": 0,
            "2": 0,
            "3": 2,
            "4": 3,
            "5": 2
            },
            "index_pos": 2,
            "width": 2,
            "fingers": 3,
            "finger_used": [1, 2, 3],
            "string": [3, 4, 5],
        },
        {
            "pitches": [
            43, 47, 50, 55, 59, 67
            ],
            "fret_config": {
            "0": 3,
            "1": 2,
            "2": 0,
            "3": 0,
            "4": 0,
            "5": 3
            },
            "index_pos": 1,
            "width": 2,
            "fingers": 3,
            "finger_used": [2, 3, 4],
            "string": [0, 1, 5],
        },
        {
            "pitches": [
            40, 47, 52, 55, 59, 64
            ],
            "fret_config": {
            "0": 0,
            "1": 2,
            "2": 2,
            "3": 0,
            "4": 0,
            "5": 0
            },
            "index_pos": 1,
            "width": 1,
            "fingers": 2,
            "finger_used": [2, 3],
            "string": [1, 2],
        },
        {
            "pitches": [
            40, 45, 52, 57, 60, 64
            ],
            "fret_config": {
            "0": 0,
            "1": 0,
            "2": 2,
            "3": 2,
            "4": 1,
            "5": 0
            },
            "index_pos": 1,
            "width": 2,
            "fingers": 3,
            "finger_used": [1, 2, 3],
            "string": [2, 3, 4],
        },

    ]

    generator.generate_and_save_forms(basic_form)