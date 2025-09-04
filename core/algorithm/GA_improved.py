import numpy as np
import random
import time
from .GA_reproduction import GAreproducing
from ..utils import set_random, visualize_guitar_tab
from remi_z import MultiTrack
from ..utils import GATab, GATabSeq
from copy import deepcopy

class GAimproved(GAreproducing):
    def __init__(self, guitar=None, 
                 mutation_rate=0.03, crossover_rate=0.6,
                 population_size=300, generations=100, 
                 num_strings=6, max_fret=20,
                 weight_PC=1.0, weight_NWC=1.0, weight_NCC=1.0,weight_RP=1.0,
                 midi_file_path=None, tournament_k=5, resolution=16):  
        self.weight_RP = weight_RP
        super().__init__(guitar, 
        mutation_rate, crossover_rate, 
        population_size, generations, num_strings, max_fret, 
        weight_PC, weight_NWC, weight_NCC, midi_file_path, tournament_k, resolution)
        
        # Add rhythm pattern functionality
        self.rhythm_pattern = None
        if midi_file_path:
            rhythm_pattern = self.extract_rhythm_pattern(midi_file_path)     
            self.rhythm_pattern = rhythm_pattern
        
        # Update statistics to include RP component
        self.statistics['fitness_stats']['avg_RP'] = 0.0
        self.statistics['ga_config']['weight_RP'] = self.weight_RP     

    def extract_rhythm_pattern(self, midi_file_path):
        """
        Extract rhythm importance pattern from MIDI file, calculated per bar.
        
        Args:
            midi_file_path: Path to the MIDI file
            
        Returns:
            List of lists: Each inner list contains importance values (0, 1, or 2) for each position in that bar
        """
        # Load MIDI file using remi_z
        mt = MultiTrack.from_midi(midi_file_path)
        
        bars_rhythm_pattern = []
        
        for bar in mt.bars:
            # Count notes at each position within this specific bar
            position_counts = {}
            
            # Get all notes in the bar (excluding drums)
            all_notes = bar.get_all_notes(include_drum=False)
            
            # Count notes at each onset position within the bar
            for note in all_notes:
                position = note.onset
                if position not in position_counts:
                    position_counts[position] = 0
                position_counts[position] += 1
            
            # Create a list with position counts for this bar
            if position_counts:
                bar_position_counts = [position_counts.get(i, 0) for i in range(self.resolution)]
            else:
                bar_position_counts = []
            
            # Calculate average note density for this bar (dynamic, adapts to different bars)
            avg_count = sum(position_counts.values()) / len(position_counts)
            
            # Classify positions by importance using this bar's average
            bar_rhythm_pattern = []
            for count in bar_position_counts:
                if count > avg_count:
                    bar_rhythm_pattern.append(2)  # Important position
                elif count > 0:
                    bar_rhythm_pattern.append(1)  # Less important but has notes
                else:
                    bar_rhythm_pattern.append(0)  # No notes
            
            bars_rhythm_pattern.append(bar_rhythm_pattern)
        
        return bars_rhythm_pattern
    
    def get_finger_assignment(self, tab_candi):
        """
        Generate finger assignments for a list of form positions.
        
        Args:
            tab_candi: List of form_pos items, where each form_pos is a list of fret positions
        
        Returns:
            List of finger assignment lists
        """
        hand_candi = []
        for form_pos in tab_candi:
            finger_assign = self.get_finger_assignment_single(form_pos)
            hand_candi.append(finger_assign)
        return hand_candi

    def get_finger_assignment_single(self, form_pos):
        """
        Generate finger assignment for a single form position.
        
        Args:
            form_pos: List of fret positions for each string (e.g., [0, 2, 0, 1, 0, 0])
        
        Returns:
            List of finger assignments for each string (-1 for unassigned/open, 1-4 for fingers)
        """
        finger_assign = [-1] * self.num_strings
        pressed = {}
        
        # Identify pressed strings
        for string_idx, fret in enumerate(form_pos):
            if fret > 0:
                pressed[string_idx] = fret
        
        if not pressed:
            return finger_assign
        
        # Group strings by fret to detect barre chords
        fret_to_strings = {}
        for string_idx, fret in pressed.items():
            if fret not in fret_to_strings:
                fret_to_strings[fret] = []
            fret_to_strings[fret].append(string_idx)
        
        # Find the smallest fret with multiple strings (barre)
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
        
        # Assign fingers
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
        
        return finger_assign
    
    def initialize_population(self):
        population=[]
        values= [-1] + list(range(self.max_fret + 1))
        fret_weight = [self.max_fret-i for i in range(self.max_fret + 1)]
        fret_weight = np.array(fret_weight,dtype=float)
        fret_weight = fret_weight / fret_weight.sum()
        raw_probabilities = [0.8]+(0.2*fret_weight).tolist()
        probabilities = np.array(raw_probabilities) / sum(raw_probabilities)
        for _ in range(self.population_size):
            tab_candi=[]
            for pos in range(self.resolution):
                if np.random.random() < 0.7:
                    form_pos = [-1]*self.num_strings
                else:
                    form_pos = np.random.choice(values, size=self.num_strings, p=probabilities).tolist()
                tab_candi.append(form_pos)
            hand_candi = self.get_finger_assignment(tab_candi)
            candi={}
            candi['tab_candi']=tab_candi
            candi['hand_candi']=hand_candi
            population.append(candi)
        return population
    def calculate_hand_pc(self, prev_hand=None, curr_hand=None, interval=None, prev_chord=None, curr_chord=None):
        hand_pc = 0
        movement_penalty = 0
        comfort_penalty = 0

        # Finger movement penalty (adapted from GAlib.py cal_finger_movement)
        if prev_hand and curr_hand and prev_chord and curr_chord:
            total_movement = 0
            
            # Build prev_finger → (string_idx, fret) mapping
            prev_finger_map = {}
            for string_idx, (prev_fret, prev_finger) in enumerate(zip(prev_chord, prev_hand)):
                if prev_finger > 0 and prev_fret > 0:
                    prev_finger_map[prev_finger] = (string_idx, prev_fret)
            
            # Check current fingers and calculate movement
            for string_idx, (curr_fret, curr_finger) in enumerate(zip(curr_chord, curr_hand)):
                if curr_finger > 0 and curr_fret > 0:
                    if curr_finger in prev_finger_map:
                        prev_string, prev_fret = prev_finger_map[curr_finger]
                        
                        # Calculate string movement
                        string_movement = abs(string_idx - prev_string)
                        
                        # Calculate fret movement  
                        fret_movement = abs(curr_fret - prev_fret)
                        
                        total_movement += string_movement + fret_movement
            
            # Apply movement penalty, adjusted by interval (longer intervals allow more movement)
            movement_penalty = -total_movement
            if interval > 6:
                movement_penalty = movement_penalty / ((interval - 6) ** 0.5)        
        # Finger comfort penalty for current position (adapted from GAlib.py cal_finger_comfort)
        if curr_hand and curr_chord:
            # Filter active fingers
            comfort_penalty = 0
            active_fingers = [(finger,string_idx, fret) for string_idx, (finger, fret) in enumerate(zip(curr_hand, curr_chord)) 
                             if finger > 0 and fret > 0]
            
            if active_fingers:
                active_fingers.sort(key=lambda x: x[0])  # Sort by finger number
                
                # Calculate fret distance penalty and string distance penalty
                fret_distance_penalty = 0
                string_distance_penalty = 0
                
                for i in range(len(active_fingers) - 1):
                    finger1, string1, fret1 = active_fingers[i]
                    finger2, string2, fret2 = active_fingers[i + 1]
                    
                    # Fret distance penalty - ideal is finger difference = fret difference
                    mismatch = fret2 - fret1 - (finger2 - finger1)
                    fret_distance_penalty += - (mismatch ** 2)
                    
                    # String distance penalty - ideal string distance should match finger distance
                    string_distance = abs(string2 - string1)
                    finger_distance = finger2 - finger1
                    if string_distance != finger_distance:
                        string_distance_penalty -= (abs(string_distance - finger_distance)) **2
                
                comfort_penalty += fret_distance_penalty + string_distance_penalty
        hand_pc = movement_penalty + comfort_penalty
        return hand_pc

    def calculate_playability(self, candidate):
        tab_candi, hand_candi = candidate['tab_candi'], candidate['hand_candi']
        active_positions = [pos for pos, chord in enumerate(tab_candi) if any(fret != -1 for fret in chord)]
        if not active_positions:
            form_pc = 0
        active_chords = [tab_candi[pos] for pos in active_positions]
        played_strings_penalty = -sum(sum(1 for fret in chord if fret > -1) for chord in active_chords)
        fret_distance_penalty = 0
        hand_pc=0
        for i in range(1, len(active_positions)):
            bar_distance_penalty = 0
            prev_chord = tab_candi[active_positions[i-1]]
            curr_chord = tab_candi[active_positions[i]]
            interval = active_positions[i] - active_positions[i-1]
            prev_frets = [f for f in prev_chord if f > 0]
            curr_frets = [f for f in curr_chord if f > 0]
            prev_hand = hand_candi[active_positions[i-1]]
            curr_hand = hand_candi[active_positions[i]]
            if prev_frets and curr_frets:
                prev_avg = sum(prev_frets) / len(prev_frets)
                curr_avg = sum(curr_frets) / len(curr_frets)
                bar_distance_penalty = abs(curr_avg - prev_avg)
            if interval >6:
                bar_distance_penalty = bar_distance_penalty / (interval-6)**0.5
            fret_distance_penalty -= bar_distance_penalty
            hand_pc += self.calculate_hand_pc(prev_hand=prev_hand, curr_hand=curr_hand, interval=interval, prev_chord=prev_chord, curr_chord=curr_chord)
        if active_positions:
            hand_pc += self.calculate_hand_pc(curr_hand=hand_candi[active_positions[0]],curr_chord=tab_candi[active_positions[0]])

        span_difficulty = 0
        for chord in active_chords:
            pressed = [fret for fret in chord if fret > 0]
            if not pressed:
                continue
            min_fret = min(pressed)
            max_fret = max(pressed)
            span = max_fret - min_fret + 1
            if span > 4:
                span_difficulty -= (span - 4) ** 2
        form_pc = (played_strings_penalty + fret_distance_penalty + span_difficulty)
        return form_pc+hand_pc
        #return form_pc
    
    def calculate_RP(self, candidate, bar_idx):
        """
        Calculate rhythm pattern fitness based on how well the arrangement matches
        the original rhythm importance pattern for the specific bar.
        
        Args:
            candidate: The candidate solution
            bar_idx: Index of the bar being processed
        """
        if not self.rhythm_pattern or bar_idx >= len(self.rhythm_pattern):
            return 0
            
        # Get the importance pattern for this specific bar
        bar_rhythm_pattern = self.rhythm_pattern[bar_idx]
        
        tab_candi, hand_candi = candidate['tab_candi'], candidate['hand_candi']
        
        # Calculate note counts for each position in the arrangement
        arrangement_counts = []
        for position in tab_candi:
            # Count how many strings are played in this position
            note_count = sum(1 for fret in position if fret > -1)
            arrangement_counts.append(note_count)
        
        # HARDCODED CLASSIFICATION for arrangement significance
        # Use fixed threshold: more than 3 strings played = significant position
        # This is reasonable for guitar since it has physical constraints (max 6 strings)
        arrangement_rhythm_pattern = []
        for count in arrangement_counts:
            if count > 3:
                arrangement_rhythm_pattern.append(2)  # Important position (dense)
            elif count > 0:
                arrangement_rhythm_pattern.append(1)  # Less important but has notes
            else:
                arrangement_rhythm_pattern.append(0)  # No notes
        
        # Calculate the differences and assign penalties
        total_penalty = 0
        for orig_imp, arr_imp in zip(bar_rhythm_pattern, arrangement_rhythm_pattern):
            diff = abs(orig_imp - arr_imp)
            if diff == 0:
                penalty = 0
            # Apply quadratic penalty for differences
            if diff == 1:
                penalty = 10
            if diff ==2:
                penalty = 100
                
            total_penalty += penalty
        
        # Convert penalty to fitness (higher is better)
        # Normalize by the ACTUAL arrangement length to be consistent with other fitness components
        fitness = -total_penalty
        
        return fitness 
    
    def calculate_NWC(self, candidate, original_midi_pitches):
        tab_candi, hand_candi = candidate['tab_candi'], candidate['hand_candi']
        total_score = 0
        for pos, (chord, targets) in enumerate(zip(tab_candi, original_midi_pitches)):
            if not targets:
                continue
            # For NWC, use the highest pitch as the main target (melody)
            target = max(targets)
            chord_dict = {j+1: fret for j, fret in enumerate(chord)}
            midi_notes = [note for note in self.guitar.get_chord_midi(chord_dict) if note != -1]
            if not midi_notes:
                total_score -= 100  # Large penalty for missing the entire position
                continue
            category = self.note_categories.get(pos, 'harmony')
            weight = self.category_weights.get(category, 1.0)
            highest_note = max(midi_notes)
            pitch_distance = abs(highest_note - target)
            if pitch_distance == 0:
                total_score += 10
            else:
                total_score -= pitch_distance ** 2 * weight 
        return total_score
    
    def calculate_NCC(self, candidate, original_midi_pitches, chord_names):
        tab_candi, hand_candi = candidate['tab_candi'], candidate['hand_candi']

        tot_err = 0
        
        # Handle both single chord name (backward compatibility) and two chord names
        if isinstance(chord_names, str):
            # Single chord for entire bar (backward compatibility)
            first_half_chord = chord_names
            second_half_chord = chord_names
        elif isinstance(chord_names, (list, tuple)) and len(chord_names) == 2:
            # Two chords: first half and second half
            first_half_chord, second_half_chord = chord_names
        else:
            # No valid chord information
            return 0
        
        # Skip if no chord names are provided
        if not first_half_chord and not second_half_chord:
            return 0
        
        # Get target chord notes for each half
        first_half_chord_notes = self.guitar.chords4NCC.get(first_half_chord, []) if first_half_chord else []
        second_half_chord_notes = self.guitar.chords4NCC.get(second_half_chord, []) if second_half_chord else []
        
        # Split bar into two halves based on resolution
        half_point = self.resolution // 2
        
        for i in range(len(tab_candi)):
            if original_midi_pitches[i]:
                target_melody = max(original_midi_pitches[i])
            else:
                target_melody = -1
            
            chord_dict = {j+1: fret for j, fret in enumerate(tab_candi[i])}
            midi_notes = self.guitar.get_chord_midi(chord_dict)
            
            # Determine which chord to use based on position
            if i < half_point:
                target_chord_notes = first_half_chord_notes
            else:
                target_chord_notes = second_half_chord_notes
            
            # Skip if no chord notes available for this half
            if not target_chord_notes:
                continue
            
            for note in midi_notes:
                if note == -1:
                    continue
                if note == target_melody:
                    continue
                note %= 12
                if note not in target_chord_notes:
                    tot_err += 1
        
        return -(tot_err)




    def fitness(self, candidate, bar_idx=None):
        original_midi_pitches = self.bars_data[bar_idx]['original_midi_pitches']
        chord_data = self.bars_data[bar_idx]['chords']
        pc = self.calculate_playability(candidate)
        nwc = self.calculate_NWC(candidate,original_midi_pitches)
        nccinrepro=super().calculate_NCC(candidate['tab_candi'],original_midi_pitches)
        ncc = self.calculate_NCC(candidate, original_midi_pitches, chord_data) if chord_data else 0
        rp = self.calculate_RP(candidate, bar_idx)
        return pc * self.weight_PC + nwc * self.weight_NWC + ncc * self.weight_NCC + rp * self.weight_RP
    
    def two_point_crossover(self, parent1, parent2):
        """
        Perform two-point crossover between two parent candidates.
        
        Args:
            parent1, parent2: Candidate dictionaries with 'tab_candi' and 'hand_candi'
        
        Returns:
            tuple: Two child candidate dictionaries
        """
        # Extract tab_candi from parent candidates
        parent1_tab = parent1['tab_candi']
        parent2_tab = parent2['tab_candi']
        
        # Perform crossover on tab_candi using parent class method
        child1_tab, child2_tab = super().two_point_crossover(parent1_tab, parent2_tab)
        
        # Generate new hand_candi for both children
        child1_hand = self.get_finger_assignment(child1_tab)
        child2_hand = self.get_finger_assignment(child2_tab)
        
        # Create new candidate dictionaries
        child1 = {'tab_candi': child1_tab, 'hand_candi': child1_hand}
        child2 = {'tab_candi': child2_tab, 'hand_candi': child2_hand}
        
        return child1, child2
    
    def mutate(self, candidate, original_midi_pitches):
        """
        Mutate a candidate by changing tab positions and regenerating finger assignments.
        
        Args:
            candidate: Candidate dictionary with 'tab_candi' and 'hand_candi'
            original_midi_pitches: Target MIDI pitches for mutation guidance
        
        Returns:
            dict: Mutated candidate dictionary
        """
        # Extract tab_candi from candidate
        tab_candi = candidate['tab_candi']
        
        # Perform mutation on tab_candi using parent class method
        mutated_tab_candi = super().mutate(tab_candi, original_midi_pitches)
        
        # Generate new hand_candi matching the mutated tab_candi
        mutated_hand_candi = self.get_finger_assignment(mutated_tab_candi)
        
        # Create new candidate dictionary
        mutated_candidate = {
            'tab_candi': mutated_tab_candi,
            'hand_candi': mutated_hand_candi
        }
        
        return mutated_candidate
    
    def run_single(self, bar_idx):
        """
        Run the genetic algorithm for a single bar.
        Args:
            bar_idx (int): Index of the bar to process.
        Returns:
            list: Best tablature found for the bar.
        """
        bar_data = self.bars_data[bar_idx]
        original_midi_pitches: list[int] = bar_data['original_midi_pitches']
        chord_data = bar_data['chords']
        population = self.initialize_population()
        best_candidate = None
        best_fitness = float('-inf')
        tournament_k = self.tournament_k  # Tournament size for selection
        for generation in range(self.generations):
            fitnesses = []
            for candidate in population:
                fit_value = self.fitness(candidate, bar_idx)
                fitnesses.append(fit_value)
            sorted_indices = np.argsort(fitnesses)[::-1]
            gen_best_idx = sorted_indices[0]
            gen_best_fitness = fitnesses[gen_best_idx]
            gen_best_candidate = population[gen_best_idx]
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_candidate = deepcopy(gen_best_candidate)
            if generation % max(1, self.generations // 5) == 0:
                pc = self.calculate_playability(gen_best_candidate)*self.weight_PC
                nwc = self.calculate_NWC(gen_best_candidate, original_midi_pitches)*self.weight_NWC
                ncc = self.calculate_NCC(gen_best_candidate, original_midi_pitches, chord_data)*self.weight_NCC
                rp = self.calculate_RP(gen_best_candidate, bar_idx)*self.weight_RP
                print(f"[Bar{bar_idx}|Gen{generation}]PC:{pc:.4f},NWC:{nwc:.4f},NCC:{ncc:.4f},RP:{rp:.4f},Total:{pc + nwc + ncc + rp:.4f}")
                print("Current best tab:")
                visualize_guitar_tab(best_candidate['tab_candi'])

            new_population = []
            new_population.append(gen_best_candidate)
            new_population.append(best_candidate)
            
            while len(new_population) < self.population_size:
                candidate_indices1 = random.sample(range(len(population)), tournament_k)
                parent1_idx = max(candidate_indices1, key=lambda idx: fitnesses[idx])
                parent1 = population[parent1_idx]
                candidate_indices2 = random.sample(range(len(population)), tournament_k)
                parent2_idx = max(candidate_indices2, key=lambda idx: fitnesses[idx])
                parent2 = population[parent2_idx]
                if random.random() < self.crossover_rate:
                    child1, child2 = self.two_point_crossover(parent1, parent2)
                else:
                    child1, child2 = deepcopy(parent1), deepcopy(parent2)
                            
                child1 = self.mutate(child1, original_midi_pitches)
                child2 = self.mutate(child2, original_midi_pitches)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            population = deepcopy(new_population)
        return best_candidate
    
    def run(self):
        """
        Run the genetic algorithm for all bars in the MIDI file.
        Returns:
            GATabSeq: Encapsulated arrangement results with onset time information.
        """
        # Start timing
        start_time = time.time()
        
        results = []
        total_pc = 0.0
        total_nwc = 0.0
        total_ncc = 0.0
        total_rp = 0.0

        
        
        for bar_idx in range(len(self.bars_data)):
            best_candidate = self.run_single(bar_idx)
            
            # 创建GATab对象，保存onset时间信息
            bar_data = self.bars_data[bar_idx]
            best_form=best_candidate['tab_candi']
            best_hand=best_candidate['hand_candi']
            ga_tab = GATab(
                bar_data=best_form,
                original_onsets=bar_data.get('original_onsets', [])
            )
            
            # 添加和弦信息
            chords = bar_data['chords']
            if len(chords) > 0:
                ga_tab.add_chord_info(0, chords[0])
            if len(chords) > 1:
                ga_tab.add_chord_info(self.resolution // 2, chords[1])
            
            # 标记旋律位置 - 从原始MIDI数据中获取旋律位置
            melody_positions = set()
            for pos, pitches in enumerate(bar_data['original_midi_pitches']):
                if pitches:  # 如果有音符在这个位置
                    # 检查这个位置是否包含旋律音符（通过比较音高）
                    melody_pitches = [pitch for pitch in self.target_melody_list[bar_idx] if pitch != -1]
                    if any(pitch in pitches for pitch in melody_pitches):
                        melody_positions.add(pos)
            
            for pos in melody_positions:
                ga_tab.add_melody_position(pos)
            
            results.append(ga_tab)
            
            # Use the structured GATab object for final fitness calculation
            pc = self.calculate_playability(best_candidate)
            nwc = self.calculate_NWC(best_candidate, self.bars_data[bar_idx]['original_midi_pitches'])
            ncc = self.calculate_NCC(best_candidate, self.bars_data[bar_idx]['original_midi_pitches'], self.bars_data[bar_idx]['chords'])
            rp = self.calculate_RP(best_candidate, bar_idx)
            print(f"[Bar {bar_idx+1}] PC: {pc:.4f}, NWC: {nwc:.4f}, NCC: {ncc:.4f}, RP: {rp:.4f}, Total: {pc + nwc + ncc + rp:.4f}")
            # Accumulate fitness statistics
            total_pc += pc
            total_nwc += nwc
            total_ncc += ncc
            total_rp += rp        
        # End timing and calculate statistics
        end_time = time.time()
        processing_time = end_time - start_time
        
        num_bars = len(self.bars_data)
        total_fitness = (total_pc * self.weight_PC + total_nwc * self.weight_NWC + 
                        total_ncc * self.weight_NCC + total_rp * self.weight_RP)
        
        # Update statistics (including RP component)
        self.statistics['processing_time_seconds'] = round(processing_time, 2)
        self.statistics['fitness_stats']['avg_PC'] = round(total_pc / num_bars, 4)
        self.statistics['fitness_stats']['avg_NWC'] = round(total_nwc / num_bars, 4)
        self.statistics['fitness_stats']['avg_NCC'] = round(total_ncc / num_bars, 4)
        self.statistics['fitness_stats']['avg_RP'] = round(total_rp / num_bars, 4)
        self.statistics['fitness_stats']['avg_fitness'] = round(total_fitness / num_bars, 4)
        
        # Return GATabSeq object - the structured approach
        ga_tab_seq = GATabSeq(results)
        return ga_tab_seq