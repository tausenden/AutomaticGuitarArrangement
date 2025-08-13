import numpy as np
import random
from ga_reproduction import GAreproducing
from GAutils import set_random

set_random(42)

class GAimproved(GAreproducing):
    def __init__(self, guitar=None, mutation_rate=0.03, population_size=300, 
                 generations=100, num_strings=6, max_fret=20, midi_file_path=None):
        super().__init__(guitar, mutation_rate, population_size, generations, num_strings, max_fret, midi_file_path)
    
    def initialize_population(self, bar_idx):
        population=[]
        values= [-1] + list(range(self.max_fret + 1))
        raw_probabilities = [0.8]+[0.2/self.max_fret for _ in range(self.max_fret + 1)]
        probabilities = np.array(raw_probabilities) / sum(raw_probabilities)
        for _ in range(self.population_size):
            form_candi=[]
            hand_candi=[]
            for pos in range(48):
                form_pos = np.random.choice(values, size=self.num_strings, p=probabilities).tolist()
                finger_assign=[-1]*self.num_strings
                pressed={}
                for string_idx,fret in enumerate(form_pos):
                    if fret>0:
                        pressed[string_idx]=fret
                if pressed:
                    fret_to_strings = {}
                    for string_idx, fret in pressed.items():
                        if fret not in fret_to_strings:
                            fret_to_strings[fret] = []
                        fret_to_strings[fret].append(string_idx)
                    
                    # Find the smallest fret with multiple strings
                    smallest_barre_fret = None
                    barre_strings = []
                    for fret in sorted(fret_to_strings.keys()):
                        if len(fret_to_strings[fret]) > 1:
                            smallest_barre_fret = fret
                            barre_strings = fret_to_strings[fret]
                            break
                    
                    # Reduce strings if more than 4 are pressed
                    all_pressed_strings = list(pressed.keys())
                    if len(all_pressed_strings) > 4:
                        # Keep strings with smaller indices, prioritizing barre strings
                        if barre_strings:
                            # Keep all barre strings and fill remaining slots with smallest indices
                            remaining_slots = 4 - 1  # barre uses 1 finger, so 3 fingers left
                            other_strings = [s for s in all_pressed_strings if s not in barre_strings]
                            other_strings.sort()  # Sort by string index
                            selected_strings = barre_strings + other_strings[:remaining_slots]
                        else:
                            # No barre, just take 4 strings with smallest indices
                            all_pressed_strings.sort()
                            selected_strings = all_pressed_strings[:4]
                    else:
                        selected_strings = all_pressed_strings
                    
                    # Step 3: Assign fingers
                    idle_fingers = [i for i in range(1, 5)]
                    random.shuffle(idle_fingers)
                    finger_idx = 0
                    
                    # If we have a barre, assign the same finger to all barre strings
                    if smallest_barre_fret is not None and any(s in selected_strings for s in barre_strings):
                        barre_finger = idle_fingers[finger_idx]
                        finger_idx += 1
                        for string_idx in barre_strings:
                            if string_idx in selected_strings:
                                finger_assign[string_idx] = barre_finger
                    
                    # Assign remaining fingers to non-barre strings
                    for string_idx in selected_strings:
                        if finger_assign[string_idx] == -1:  # Not assigned yet
                            if finger_idx < len(idle_fingers):
                                finger_assign[string_idx] = idle_fingers[finger_idx]
                                finger_idx += 1
                form_candi.append(form_pos)
                hand_candi.append(finger_assign)
            population.append((form_candi,hand_candi))
        return population