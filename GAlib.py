import numpy as np
import random
from utlis import Guitar
from utlis import pitch2name,visualize_guitar_tab

class GuitarGeneticAlgorithm:
    def __init__(self, target_melody, target_chords=None, guitar=None, 
                 mutation_rate=0.3, population_size=1000, generations=800, 
                 tournament_size=None, reserved_ratio=0.2, num_strings=6, 
                 max_fret=8, num_fingers=4, w_PC=1.0, w_NWC=1.0, w_NCC=1.0):
        """
        如果传入的是list of list，则认为是多个目标旋律，否则为单个目标旋律。
        """
        if(guitar==None):
            assert('no guitar wrong')

        self.guitar = guitar
        # 判断传入是否为 list of list（至少包含一个列表）
        if isinstance(target_melody, list) and target_melody and isinstance(target_melody[0], list):
            self.target_melody_list = target_melody
        else:
            self.target_melody_list = [target_melody]

        if isinstance(target_chords, list) and target_chords:
            self.target_chord_list = target_chords
        else:
            self.target_chord_list = [target_chords]
            
        # Algorithm parameters
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = population_size if tournament_size == None else tournament_size
        self.reserved = int(population_size * reserved_ratio)
        self.num_strings = num_strings
        self.max_fret = max_fret
        self.num_fingers = num_fingers
        
        # Weights for fitness calculation
        self.w_PC = w_PC
        self.w_NWC = w_NWC
        self.w_NCC = w_NCC

    def get_melody_pitch(self, chord_fingering):
        """从六元组中提取最高音的MIDI pitch"""
        fingering_dict = {i+1: fret for i, fret in enumerate(chord_fingering)}
        midi_notes = self.guitar.get_chord_midi(fingering_dict)
        return max(midi_notes) if midi_notes else None
    
    def cal_PC(self, sequence,verbose=0):

        PC1 = 0
        for chord in sequence:
            # Count notes actually being played (fret > 0, not open strings)
            notes_played = 0
            for fret in chord:
                if fret > 0:  # Only count actual fret presses
                    notes_played += 1
            
            # Base penalty for each note played
            chord_penalty = -notes_played
            
            # Additional penalty for playing more than 3 notes
            if notes_played > 3:
                chord_penalty -= (notes_played+3) **2
            
            PC1 += chord_penalty    

        PC2 = -sum((max(chord) - max(min(chord), 0)) for chord in sequence)  # width of the press

        PC3 = 0 #avg press position
        for chord in sequence:
            fretted_notes = [fret for fret in chord if fret > 0]
            avg_fret = sum(fretted_notes) / max(1, len(fretted_notes)) if fretted_notes else 0
            PC3 -= avg_fret

        PC4 = 0 # fret changes
        for i in range(1, len(sequence)):
            prev_chord = sequence[i - 1]
            curr_chord = sequence[i]
            hand_movement = sum(abs(curr_fret - prev_fret) 
                                for curr_fret, prev_fret in zip(curr_chord, prev_chord) 
                                if curr_fret > 0 and prev_fret > 0)
            PC4 -= hand_movement
        if verbose:
            print("string press in the same time",PC1)
            print("width of the press",PC2)
            print("avg press position",PC3)
            print("fret changes",PC4)
        return (PC1 + PC2 + PC3 + PC4)/len(sequence)
    
    def cal_NWC(self, sequence, target_melody):
        total_error = 0
        valid_notes = 0

        for i in range(min(len(sequence), len(target_melody))):
            if target_melody[i] == -1:
                continue

            chord_dict = {i+1: fret for i, fret in enumerate(sequence[i])}
            midi_notes = self.guitar.get_chord_midi(chord_dict)
            
            if not midi_notes:  # 如果和弦没有音符
                total_error += 100  # 惩罚空和弦
                continue
                
            # 找到最接近目标音高的音符
            # position 预测，结合guitar类
            target_pitch = target_melody[i]

            # if target_pitch == -1:
            #     for note in midi_notes:
            #         if note != -1:
            #             total_error += 100
            
            # comment because -1 doesn't mean no note is played

            # closest_pitch_error = min(abs(note - target_pitch) for note in midi_notes)
            
            # # 计算误差并加权
            # pitch_error = closest_pitch_error
            # if max(midi_notes) != target_pitch:  # 如果最高音不是目标音高
            #     pitch_error += 10  # 额外惩罚

            pitch_error=abs(max(midi_notes)-target_pitch)
                
            total_error += pitch_error
            valid_notes += 1
        return -(total_error)/len(sequence)

    def cal_NCC(self, play_seq, target_melody, chord_name, verbose=0): # how many notes not in the triad of chord
        tot_err=0
        if not chord_name:
            return 0
    
        target_chord_note=self.guitar.chords4NCC[chord_name]

        if verbose:
            print("chord_name",chord_name)
            print("target_chord_note",target_chord_note)

        for i in range(len(play_seq)):
            chord_dict = {i+1: fret for i, fret in enumerate(play_seq[i])}
            midi_notes = self.guitar.get_chord_midi(chord_dict)

            if verbose:
                print("chord_dict",chord_dict)
                print("midi_notes",midi_notes)

            for note in midi_notes:
                if note == -1:
                    continue
                if note != -1 and note == target_melody[i]:
                    continue
                
                note%=12
                if note!=-1 and note not in target_chord_note:
                    tot_err+=1
                    
        return -(tot_err)/len(play_seq)
        
    def fitness(self, sequence, target_melody, target_chord):
        PC = self.cal_PC(sequence)
        NWC = self.cal_NWC(sequence, target_melody)
        NCC = self.cal_NCC(sequence, target_melody, target_chord)
        #print("PC,NWC,NCC",PC,NWC,NCC)
        fitness_value = PC * self.w_PC + NWC * self.w_NWC + NCC * self.w_NCC
        return fitness_value
    
    def initialize_population(self, target_melody):
        """初始化种群，每个个体为一个序列（list of chords），个体长度与目标旋律相同"""
        population = []
        
        values = [-1] + list(range(self.max_fret + 1))
        raw_probabilities = [0.9] + [0.1 * (self.max_fret - i + 1) for i in range(self.max_fret + 1)]
        probabilities = np.array(raw_probabilities) / sum(raw_probabilities)
        
        for _ in range(self.population_size):
            sequence = []
            for _ in range(len(target_melody)):
                chord = np.random.choice(values, size=self.num_strings, p=probabilities).tolist()
                sequence.append(chord)
            population.append(sequence)
        
        return population

    def tournament_selection(self, population, fitnesses):
        """锦标赛选择：随机选取部分个体，并根据适应度排序"""
        candidates_idx = random.sample(range(len(population)), min(self.tournament_size, len(population)))
        candidates_idx.sort(key=lambda idx: fitnesses[idx], reverse=True)
        sorted_candidates = [population[idx] for idx in candidates_idx]
        return sorted_candidates

    def crossover(self, parent1, parent2):
        """交叉操作：在随机两个点之间交换片段"""
        if len(parent1) < 2:
            return parent1.copy()
        points = sorted(random.sample(range(len(parent1)), 2))
        child = parent1[:points[0]] + parent2[points[0]:points[1]] + parent1[points[1]:]
        return child

    def mutate(self, sequence):
        """变异操作，对每个和弦中的每个音符以 mutation_rate 概率随机变异"""
        mutated_sequence = []
        for chord in sequence:
            mutated_chord = []
            for note in chord:
                if random.random() < self.mutation_rate:
                    mutated_chord.append(random.randint(-1, self.max_fret))
                else:
                    mutated_chord.append(note)
            mutated_sequence.append(mutated_chord)
        return mutated_sequence

    def run_single(self, target_melody, target_chord):
        """对单个目标旋律运行遗传算法"""
        population = self.initialize_population(target_melody)

        for generation in range(self.generations):
            fitnesses = [self.fitness(ind, target_melody, target_chord) for ind in population]
            best_fit = max(fitnesses)
            best_sequence = population[fitnesses.index(best_fit)]

            if generation % max(1, (self.generations // 10)) == 0:
                melody_pitch = [self.get_melody_pitch(chord) for chord in best_sequence]
                print(f"Generation {generation}: Best Fitness = {best_fit}")
                print("Best sequence melody:", pitch2name(melody_pitch))
                print("Best finger position:", best_sequence)

            candidates = self.tournament_selection(population, fitnesses)
            new_population = []
            candidate_idx = 0
            for _ in range(self.population_size - self.reserved):
                candidate_idx %= (len(candidates) - 1)
                parent1 = candidates[candidate_idx]
                parent2 = candidates[candidate_idx + 1]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
                candidate_idx += 1
            population = new_population
            for i in range(self.reserved):
                population.append(candidates[i])

        final_fitnesses = [self.fitness(ind, target_melody, target_chord) for ind in population]
        best_sequence = population[np.argmax(final_fitnesses)]
        return best_sequence

    def run(self):
        """
        如果目标旋律是 list of list，则依次对每个旋律运行遗传算法，
        返回一个包含各个旋律最佳序列的列表；否则返回单个最佳序列。
        """
        results = []
        for i in range(len(self.target_melody_list)):
            target_melody = self.target_melody_list[i]
            target_chord = self.target_chord_list[i]
            print("\nProcessing target melody:", target_melody, target_chord)
            best_sequence = self.run_single(target_melody, target_chord)
            results.append(best_sequence)
        # 如果原始输入为单个旋律，则直接返回该序列，否则返回列表
        if len(self.target_melody_list) == 1:
            return results[0]
        else:
            return results


class HandGuitarGA(GuitarGeneticAlgorithm):
    def __init__(self, target_melody, target_chords=None, guitar=None,
                 mutation_rate=0.3, population_size=1000, generations=800, 
                 tournament_size=1000, reserved_ratio=0.2, num_strings=6, 
                 max_fret=8, num_fingers=4, w_PC=0.0, w_NWC=0.0, w_NCC=1.0):
        super().__init__(
            target_melody, target_chords, guitar,
            mutation_rate, population_size, generations,
            tournament_size, reserved_ratio, num_strings,
            max_fret, num_fingers, w_PC, w_NWC, w_NCC
        )

    def get_melody_pitch(self, chord_data):
        """从增强的和弦数据中提取最高音的MIDI pitch"""
        fingering, _ = chord_data  # 解包和弦数据，忽略手指信息
        return super().get_melody_pitch(fingering)

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

    def cal_PC(self, sequence,verbose=0):

        fingerings, finger_assignments = zip(*sequence)
        PC1_4=super().cal_PC(fingerings,verbose)

        PC5 = sum(self.cal_finger_comfort(fingerings[i], finger_assignments[i]) for i in range(len(fingerings)))

        PC6 = 0
        for i in range(1, len(finger_assignments)):
            PC6 += self.cal_finger_movement(fingerings[i - 1], finger_assignments[i - 1], fingerings[i], finger_assignments[i])


        PC7 = 0
        for i in range(len(finger_assignments)):
            PC7+=self.cal_finger_order(fingerings[i],finger_assignments[i])

        if verbose:
            print("finger_comfort",PC5)
            print("finger_movement",PC6)
            print("finge_order",PC7)

        return PC1_4 + (PC5 + PC6 + PC7) / len(sequence)
    
    
    def cal_NWC(self, sequence, target_melody):
        fingerings, _ = zip(*sequence)
        return super().cal_NWC(fingerings, target_melody)

    def cal_NCC(self, play_seq, target_melody,chord_name,verbose=0):
        fingerings, _ = zip(*play_seq)
        return super().cal_NCC(fingerings, target_melody, chord_name,verbose)

    def initialize_population(self, target_melody):
        """初始化种群，现在每个和弦包含按弦位置和手指分配"""

        population = super().initialize_population(target_melody)
        
        for sequence in population:
            for chord in sequence:
                finger_assignment = [-1] * self.num_strings
                pressed_strings = [i for i, fret in enumerate(chord) if fret > 0]

                if pressed_strings:
                    available_fingers = list(range(1, self.num_fingers + 1))
                    random.shuffle(available_fingers)
                    for i, string_idx in enumerate(pressed_strings):
                        if i < len(available_fingers):
                            finger_assignment[string_idx] = available_fingers[i]

                # 替换原来的和弦数据，使其包含手指分配
                sequence[sequence.index(chord)] = (chord, finger_assignment)
        
        return population

    def mutate1(self, sequence):
        """变异操作，同时处理按弦位置和手指分配"""
        mutated_sequence = []
        for fingering, finger_assignment in sequence:
            if random.random() < self.mutation_rate:
                mutated_fingering = [
                    random.randint(-1, self.max_fret) for fret in fingering
                ]
            else:
                mutated_fingering = fingering

            mutated_fingers = [-1] * self.num_strings
            pressed_strings = [i for i, fret in enumerate(mutated_fingering) if fret > 0]

            if pressed_strings:
                available_fingers = list(range(1, self.num_fingers + 1))
                random.shuffle(available_fingers)
                for i, string_idx in enumerate(pressed_strings):
                    if i < len(available_fingers):
                        mutated_fingers[string_idx] = available_fingers[i]

            mutated_sequence.append((mutated_fingering, mutated_fingers))

        return mutated_sequence
    def mutate2(self, sequence, target_melody, target_chord):
        """Mutate the sequence with melody and chord awareness."""
        mutated_sequence = []
        chord_notes = self.guitar.chords4NCC[target_chord]  # Get chord notes in MIDI
        
        for idx, (fingering, finger_assignment) in enumerate(sequence):
            # Create a new fingering for this position
            mutated_fingering = []
            
            # Process each string
            for string_idx, fret in enumerate(fingering):
                original_pitch = self.guitar.fboard[string_idx + 1][fret] if fret >= 0 else -1
                mutate_prob = self.mutation_rate
                
                # Adjust mutation probability based on melody/chord
                if original_pitch in [target_melody[idx]] or (original_pitch % 12) in chord_notes:
                    mutate_prob /= 2  # Reduce mutation probability
                else:
                    mutate_prob *= 2  # Increase mutation probability for non-melodic, non-chord tones
                
                # Decide whether to mutate this string
                if random.random() < mutate_prob:
                    # Create a collection of valid frets
                    valid_frets = []
                    
                    # Add open/not pressed string as an option
                    valid_frets.append(-1)

                    # Add frets that produce melody notes or chord tones
                    if target_melody[idx] == -1:
                        for f in range(1, self.max_fret + 1):
                            fret_pitch = self.guitar.fboard[string_idx + 1][f]
                            if (fret_pitch % 12) in chord_notes:
                                valid_frets.append(f)
                    else:
                        for f in range(1, self.max_fret + 1):
                            fret_pitch = self.guitar.fboard[string_idx + 1][f]
                            if fret_pitch in [target_melody[idx], target_melody[idx] + 12, target_melody[idx] - 12] or (fret_pitch % 12) in chord_notes:
                                valid_frets.append(f)
                    
                    # If no valid frets found that match melody/chord, use other frets
                    if len(valid_frets) == 0:  # Only -1 is in the list
                        valid_frets.extend(range(-1, self.max_fret + 1))
                    
                    # Choose a fret randomly from the valid options
                    mutated_fingering.append(random.choice(valid_frets))
                else:
                    # Keep the original fret
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
        
        return mutated_sequence

    def fitness(self, sequence, target_melody, target_chord):
        """计算适应度"""
        PC = self.cal_PC(sequence)
        NWC = self.cal_NWC(sequence, target_melody)
        NCC = self.cal_NCC(sequence, target_melody, target_chord)

        return PC * self.w_PC + NWC * self.w_NWC + NCC * self.w_NCC

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
                #print("Best sequence fingerings:", best_sequence)
                print("NCC now",self.cal_NCC(best_sequence,target_melody,target_chord)*len(best_sequence))
                print("NWC now",self.cal_NWC(best_sequence,target_melody)*len(best_sequence))
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
                child = self.mutate2(child, target_melody, target_chord)
                #child = self.mutate1(child)
                new_population.append(child)
                candidate_idx += 1

            population = new_population + candidates[:self.reserved]

        final_fitnesses = [self.fitness(ind, target_melody, target_chord) for ind in population]
        best_sequence = population[np.argmax(final_fitnesses)]
        return best_sequence