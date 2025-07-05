import numpy as np
import random
from GAutils import Guitar
from GAutils import pitch2name, visualize_guitar_tab
from GAlib import GuitarGeneticAlgorithm
from remi_z import MultiTrack

def extract_rhythm_importance(midi_file_path):
    
    # Load MIDI file using remi_z
    mt = MultiTrack.from_midi(midi_file_path)
    
    # Count notes at each position across all bars
    position_counts = {}
    
    for bar in mt.bars:
        # Get all notes in the bar (excluding drums)
        all_notes = bar.get_all_notes(include_drum=False)
        
        # Count notes at each onset position within the bar
        for note in all_notes:
            position = note.onset
            if position not in position_counts:
                position_counts[position] = 0
            position_counts[position] += 1
    
    # Create a list with position counts
    if position_counts:
        max_position = max(position_counts.keys())
        all_position_counts = [position_counts.get(i, 0) for i in range(max_position + 1)]
    else:
        all_position_counts = []
    
    # Calculate average note density (dynamic, adapts to different songs)
    avg_count = sum(all_position_counts) / len(all_position_counts) if all_position_counts else 0
    
    # Classify positions by importance using dynamic average
    importance = []
    for count in all_position_counts:
        if count > avg_count:
            importance.append(2)  # Important position
        elif count > 0:
            importance.append(1)  # Less important but has notes
        else:
            importance.append(0)  # No notes
    
    return importance

class HandGuitarGA(GuitarGeneticAlgorithm):
    def __init__(self, target_melody, target_chords=None, guitar=None, midi_file_path=None,
             mutation_rate=0.3, population_size=1000, generations=800, 
             tournament_size=1000, reserved_ratio=0.2, num_strings=6, 
             max_fret=8, num_fingers=4, w_PC=1.0, w_NWC=1.0, w_NCC=1.0, w_RP=1.0):
        super().__init__(
            target_melody, target_chords, guitar,
            mutation_rate, population_size, generations,
            tournament_size, reserved_ratio, num_strings,
            max_fret, num_fingers, w_PC, w_NWC, w_NCC
        )
        self.w_RP = w_RP
        self.rhythm_importance = None
        if midi_file_path:
            self.rhythm_importance = extract_rhythm_importance(midi_file_path)

    def get_melody_pitch(self, chord_data):
        """从增强的和弦数据中提取最高音的MIDI pitch"""
        fingering, _ = chord_data  # 解包和弦数据，忽略手指信息
        return super().get_melody_pitch(fingering)
    
    def calculate_rhythm_pattern_fitness(self, arrangement, original_importance):
        """
        Calculate the rhythm pattern fitness by comparing the importance of positions 
        in the original MIDI and the arrangement.
        Now uses HARD CLASSIFICATION for arrangement: more than 3 notes = significant.
        
        Parameters:
        -----------
        arrangement : list of lists or tuples
            The guitar arrangement, where each element represents a chord position
        original_importance : list
            The importance classification of positions from the original MIDI
            
        Returns:
        --------
        float
            The rhythm pattern fitness score (higher is better)
        """
        # Count notes in arrangement at each position
        arrangement_counts = []
        
        for position in arrangement:
            # Handle different types of arrangement data structures
            if isinstance(position, tuple):
                # If arrangement contains finger positions as well
                fingering, _ = position
            else:
                fingering = position
                
            # Count how many strings are played in this position
            note_count = sum(1 for fret in fingering if fret > -1)
            arrangement_counts.append(note_count)
        
        # HARD CLASSIFICATION for arrangement significance
        # Use fixed threshold: more than 3 notes played = significant position
        arrangement_importance = []
        for count in arrangement_counts:
            if count > 3:
                arrangement_importance.append(2)  # Important position (dense)
            elif count > 0:
                arrangement_importance.append(1)  # Less important but has notes
            else:
                arrangement_importance.append(0)  # No notes
        
        # Make sure we're comparing equal length sequences
        # Use the arrangement length (which should match target_melody length)
        comparison_length = min(len(original_importance), len(arrangement_importance))
        original_importance_trimmed = original_importance[:comparison_length]
        arrangement_importance_trimmed = arrangement_importance[:comparison_length]
        
        # Calculate the differences and assign penalties
        total_penalty = 0
        for orig_imp, arr_imp in zip(original_importance_trimmed, arrangement_importance_trimmed):
            diff = abs(orig_imp - arr_imp)
            
            # Apply quadratic penalty for differences
            penalty = diff * diff
                
            total_penalty += penalty
        
        # Convert penalty to fitness (higher is better)
        # Normalize by the ACTUAL arrangement length to be consistent with other fitness components
        fitness = -total_penalty / len(arrangement)
        
        return fitness
    
    def cal_finger_comfort(self, fret_positions, finger_positions):
        """计算手指位置的舒适度，结合弦之间的横向距离和品位之间的纵向距离"""
        if not fret_positions or not finger_positions:
            return 0

        # 过滤出有效的 (弦号, 手指编号, 品位)
        active_fingers = [(i, finger, fret) for i, (finger, fret) in enumerate(zip(finger_positions, fret_positions)) if finger > 0 and fret > 0]
        if not active_fingers:
            return 0

        active_fingers.sort(key=lambda x: x[1])  # 依据手指编号排序

        # 计算相邻手指的品位间距 & 弦间距
        fret_distance_penalty = 0
        string_distance_penalty = 0

        for i in range(len(active_fingers) - 1):
            string1, finger1, fret1 = active_fingers[i]
            string2, finger2, fret2 = active_fingers[i + 1]

            # 计算纵向（品位）间距，理想情况是 2
            fret_distance_penalty += min(0,-abs(fret2 - fret1-(finger2-finger1)))

            # 计算横向（弦）间距，理想情况是相邻或隔 1 弦
            string_distance = abs(string2 - string1)
            if string_distance == 1:
                string_distance_penalty += 0  # 相邻弦，不惩罚
            elif string_distance == 2:
                string_distance_penalty += -1  # 隔 1 弦，轻微惩罚
            else:
                string_distance_penalty += -((string_distance-(finger2-finger1-1))**2)

        return fret_distance_penalty + string_distance_penalty

    def cal_finger_movement(self, prev_frets, prev_fingers, curr_frets, curr_fingers):
        """计算手指在两个连续和弦之间的移动量，包括品位移动和弦移动"""
        total_movement = 0
        
        # 构建 prev_finger → (弦索引, 品位) 映射
        prev_finger_map = {}  
        for string_idx, (prev_fret, prev_finger) in enumerate(zip(prev_frets, prev_fingers)):
            if prev_finger > 0 and prev_fret > 0:  
                prev_finger_map[prev_finger] = (string_idx, prev_fret)  # 记录手指在哪根弦、哪个品位

        # 遍历 curr_fingers，找到它在 prev_fingers 里的位置
        for string_idx, (curr_fret, curr_finger) in enumerate(zip(curr_frets, curr_fingers)):
            if curr_finger > 0 and curr_fret > 0:  
                if curr_finger in prev_finger_map:
                    prev_string, prev_fret = prev_finger_map[curr_finger]

                    # 计算弦的移动量
                    string_movement = abs(string_idx - prev_string)

                    # 计算品位的移动量
                    fret_movement = abs(curr_fret - prev_fret)

                    total_movement += string_movement + fret_movement
        return -total_movement  

    def cal_finger_order(self, fret_positions, finger_positions):
        active_fingers = [(finger, fret) for finger, fret in zip(finger_positions, fret_positions) if finger > 0 and fret > 0]
        
        penalty = 0

        for i in range(len(active_fingers) - 1):
            finger1, fret1 = active_fingers[i]
            finger2, fret2 = active_fingers[i + 1]
            if finger2 > finger1 and fret2 < fret1:  
                penalty -= finger2 - finger1 + fret1 - fret2  # 惩罚手指交叉
        return penalty

    def cal_PC(self, sequence, verbose=0):
        fingerings, finger_assignments = zip(*sequence)
        
        # Call parent class PC calculation (note density penalty)
        PC1_4 = super().cal_PC(fingerings, verbose)

        # Add finger-specific penalties
        PC5 = sum(self.cal_finger_comfort(fingerings[i], finger_assignments[i]) for i in range(len(fingerings)))

        PC6 = 0
        for i in range(1, len(finger_assignments)):
            PC6 += self.cal_finger_movement(fingerings[i - 1], finger_assignments[i - 1], 
                                          fingerings[i], finger_assignments[i])

        PC7 = sum(self.cal_finger_order(fingerings[i], finger_assignments[i]) for i in range(len(fingerings)))

        return PC1_4 + PC5 + PC6 + PC7

    def mutate1(self, sequence):
        """Enhanced mutation for finger assignment sequences"""
        mutated_sequence = []
        
        for fingering, fingers in sequence:
            if random.random() < self.mutation_rate:
                # Mutate the fingering
                mutated_fingering = []
                for i, fret in enumerate(fingering):
                    if random.random() < 0.3:  # 30% chance to mutate each string
                        valid_frets = list(range(-1, self.max_fret + 1))
                        mutated_fingering.append(random.choice(valid_frets))
                    else:
                        mutated_fingering.append(fret)
                
                # Assign fingers to pressed strings
                mutated_fingers = [-1] * self.num_strings
                pressed_strings = [i for i, fret in enumerate(mutated_fingering) if fret > 0]
                
                if pressed_strings:
                    available_fingers = list(range(1, self.num_fingers + 1))
                    random.shuffle(available_fingers)
                    for i, string_idx in enumerate(pressed_strings):
                        if i < len(available_fingers):
                            mutated_fingers[string_idx] = available_fingers[i]
                
                mutated_sequence.append((mutated_fingering, mutated_fingers))
            else:
                mutated_sequence.append((fingering, fingers))
        
        return mutated_sequence

    def fitness(self, sequence, target_melody, target_chord):
        """Calculate the overall fitness including rhythm pattern."""
        PC = self.cal_PC(sequence)
        NWC = self.cal_NWC(sequence, target_melody)
        NCC = self.cal_NCC(sequence, target_melody, target_chord)
        
        # Include rhythm pattern fitness if available
        RP = 0
        if self.rhythm_importance:
            RP = self.calculate_rhythm_pattern_fitness(sequence, self.rhythm_importance)
        
        fitness_value = (PC * self.w_PC + 
                        NWC * self.w_NWC + 
                        NCC * self.w_NCC + 
                        RP * self.w_RP)
        
        return fitness_value

    def run_single(self, target_melody, target_chord):
        """运行单个目标旋律的遗传算法"""
        population = self.initialize_population(target_melody)
        
        for generation in range(self.generations):
            fitnesses = [self.fitness(ind, target_melody, target_chord) for ind in population]
            best_fit = max(fitnesses)
            best_sequence = population[fitnesses.index(best_fit)]

            if generation % max(1, (self.generations // 10)) == 0:
                melody_pitch = [self.get_melody_pitch(chord_data) for chord_data in best_sequence]
                print(f"Generation {generation}: Best Fitness = {best_fit}")
                print("Best sequence melody:", pitch2name(melody_pitch))
                print("PC now", self.cal_PC(best_sequence)*len(best_sequence))
                print("NCC now",self.cal_NCC(best_sequence,target_melody,target_chord)*len(best_sequence))
                print("NWC now",self.cal_NWC(best_sequence,target_melody)*len(best_sequence))
                print("RP now",self.calculate_rhythm_pattern_fitness(best_sequence, self.rhythm_importance)*len(best_sequence))
                fret, finger = zip(*best_sequence)
                print(fret)
                visualize_guitar_tab(fret)

            if generation == self.generations - 1:
                result = self.cal_NCC(best_sequence, target_melody,target_chord)
                print('notes not in chord', result * len(best_sequence))

            candidates = self.tournament_selection(population, fitnesses)
            new_population = []
            candidate_idx = 0

            for _ in range(self.population_size - self.reserved):
                candidate_idx %= (len(candidates) - 1)
                parent1 = candidates[candidate_idx]
                parent2 = candidates[candidate_idx + 1]
                child = self.crossover(parent1, parent2)
                child = self.mutate1(child)
                new_population.append(child)
                candidate_idx += 1

            population = new_population + candidates[:self.reserved]

        final_fitnesses = [self.fitness(ind, target_melody, target_chord) for ind in population]
        best_sequence = population[np.argmax(final_fitnesses)]
        return best_sequence