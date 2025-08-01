import json

class JSONLToFormsConverter:
    def __init__(self):
        self.open_strings = [40, 45, 50, 55, 59, 64]
        
    def convert_tab_to_fret_config(self, tab):
        fret_config = {}
        for i, fret in enumerate(tab):
            if fret == "x":
                fret_config[str(i)] = -1
            else:
                fret_config[str(i)] = int(fret)
        return fret_config
    
    def calculate_pitches_from_tab(self, tab):
        pitches = []
        for string_idx, fret in enumerate(tab):
            if fret != "x":
                pitch = self.open_strings[string_idx] + int(fret)
                pitches.append(pitch)
        return sorted(pitches)
    
    def calculate_form_properties(self, tab, fingering):
        pressed_frets = []
        active_strings = []
        finger_positions = []
        
        for i, (fret, finger) in enumerate(zip(tab, fingering)):
            if fret != "x" and fret != 0:
                pressed_frets.append(int(fret))
                active_strings.append(i)
                if finger > 0:
                    finger_positions.append(finger)
        
        if not pressed_frets:
            index_pos = 0
        else:
            min_fret = min(pressed_frets)
            index_pos = max(1, min_fret - 1) if min_fret > 1 else min_fret
        
        if not pressed_frets:
            width = 0
        else:
            width = max(pressed_frets) - min(pressed_frets) + 1
        
        fingers_used = [f for f in finger_positions if f > 0]
        n_fingers = len(set(fingers_used))
        
        strings_used = []
        for i, fret in enumerate(tab):
            if fret != "x":
                strings_used.append(i)
        
        return {
            'index_pos': index_pos,
            'width': width,
            'fingers': n_fingers,
            'finger_used': sorted(list(set(fingers_used))) if fingers_used else [],
            'string': strings_used
        }
    
    def convert_entry(self, entry):
        tab = entry['tab']
        fingering = entry.get('fingering', [0] * len(tab))
        
        pitches = self.calculate_pitches_from_tab(tab)
        fret_config = self.convert_tab_to_fret_config(tab)
        props = self.calculate_form_properties(tab, fingering)
        
        return {
            "pitches": pitches,
            "fret_config": fret_config,
            "index_pos": props['index_pos'],
            "width": props['width'],
            "fingers": props['fingers'],
            "finger_used": props['finger_used'],
            "string": props['string']
        }
    
    def convert_file(self, input_file, output_file):
        with open(input_file, 'r') as f:
            content = f.read()
        
        forms = []
        
        if content.strip().startswith('['):
            data = json.loads(content)
            for entry in data:
                forms.append(self.convert_entry(entry))
        else:
            for line in content.strip().split('\n'):
                if line.strip():
                    entry = json.loads(line)
                    forms.append(self.convert_entry(entry))
        
        with open(output_file, 'w') as f:
            json.dump(forms, f, indent=2)
        
        return forms
def main():
    converter = JSONLToFormsConverter()
    input_file = 'possible_positions.jsonl'
    output_file = 'guitar_forms_CAGED.json'
    forms = converter.convert_file(input_file, output_file)  # or input.json   

if __name__ == "__main__":
    main()