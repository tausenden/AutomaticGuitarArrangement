import numpy as np
import random
from utlis import Guitar
from utlis import pitch2name

MUTATION_RATE = 0.3
POPULATION_SIZE = 1000
GENERATIONS = 800
TOURNAMENT_SIZE = 1000
RESERVED= POPULATION_SIZE//5
NUM_STRINGS = 6  # 吉他弦数
MAX_FRET = 8    # 最大品格数
NUM_FINGERS = 4  # 不包括大拇指

class GuitarGeneticAlgorithm:
    def __init__(self, target_melody, target_chords=None ,guitar=None):
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

    def get_melody_pitch(self, chord_fingering):
        """从六元组中提取最高音的MIDI pitch"""
        fingering_dict = {i+1: fret for i, fret in enumerate(chord_fingering)}
        midi_notes = self.guitar.get_chord_midi(fingering_dict)
        return max(midi_notes) if midi_notes else None
    
    def cal_PC(self, sequence):
        PC1 = -sum(sum(1 for fret in chord if fret > 0) for chord in sequence)  # string press in the same time
        PC2 = -sum((max(chord) - max(min(chord), 0)) for chord in sequence)  # width of the press

        total_avg_fret = 0
        for chord in sequence:
            fretted_notes = [fret for fret in chord if fret > 0]
            avg_fret = sum(fretted_notes) / max(1, len(fretted_notes)) if fretted_notes else 0
            total_avg_fret += avg_fret

        PC3 = total_avg_fret  # average fret press

        PC4 = 0
        for i in range(1, len(sequence)):
            prev_chord = sequence[i - 1]
            curr_chord = sequence[i]
            hand_movement = sum(abs(curr_fret - prev_fret) for curr_fret, prev_fret in zip(curr_chord, prev_chord) if curr_fret > 0 and prev_fret > 0)
            PC4 += hand_movement
        PC4 *= -1  # hand movement
        return (PC1 + PC2 + PC3 + PC4)/len(sequence)
    
    def cal_NWC(self, sequence,target_melody):

        total_error = 0
        valid_notes = 0

        for i in range(min(len(sequence), len(target_melody))):
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
    
    def cal_NCC(self, play_seq, chord_name): # how many notes are not in chord
        tot_err=0
        if not chord_name:
            return 0
        chord_dict=self.guitar.chords[chord_name][0]
        
        chord_six=[]
        for i in range(6):
            chord_six.append(chord_dict[i+1])
        for note in play_seq:
            for i in range(6):
                tot_err+=abs(note[i]-chord_six[i])
        return -(tot_err)/len(play_seq)

    def new_cal_NCC(self, play_seq, chord_name): # how many notes not in the triad of chord
        
        tot_err=0
        target_chord_note=self.guitar.chords4NCC[chord_name]
        if not chord_name:
            return 0
        for i in range(len(play_seq)):
            chord_dict = {i+1: fret for i, fret in enumerate(play_seq[i])}
            midi_notes = self.guitar.get_chord_midi(chord_dict)
            for note in midi_notes:
                note%=12
                if note!=-1 and note not in target_chord_note:
                    tot_err+=1
                    
        return -(tot_err)/len(play_seq)
        
        

    def fitness(self, sequence, target_melody,target_chord):
        w_PC=1.0
        w_NWC=5.0
        w_NCC=1.0 

        PC=self.cal_PC(sequence)
        NWC=self.cal_NWC(sequence,target_melody)
        # NCC=self.cal_NCC(sequence,target_chord)
        NCC=self.new_cal_NCC(sequence,target_chord)
        #print("PC,NWC,NCC",PC,NWC,NCC)
        fitness_value=PC*w_PC+NWC*w_NWC+NCC*w_NCC
        return fitness_value


    def initialize_population(self, target_melody):
        """初始化种群，每个个体为一个序列（list of chords），个体长度与目标旋律相同"""
        population = []
        for _ in range(POPULATION_SIZE):
            sequence = []
            for _ in range(len(target_melody)):
                chord = [random.randint(-1, MAX_FRET) for _ in range(NUM_STRINGS)]
                sequence.append(chord)
            population.append(sequence)
        return population

    def tournament_selection(self, population, fitnesses):
        """锦标赛选择：随机选取部分个体，并根据适应度排序"""
        candidates_idx = random.sample(range(len(population)), TOURNAMENT_SIZE)
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
        """变异操作，对每个和弦中的每个音符以 MUTATION_RATE 概率随机变异"""
        mutated_sequence = []
        for chord in sequence:
            mutated_chord = []
            for note in chord:
                if random.random() < MUTATION_RATE:
                    mutated_chord.append(random.randint(-1, MAX_FRET))
                else:
                    mutated_chord.append(note)
            mutated_sequence.append(mutated_chord)
        return mutated_sequence

    def run_single(self, target_melody, target_chord):
        """对单个目标旋律运行遗传算法"""
        population = self.initialize_population(target_melody)

        for generation in range(GENERATIONS):
            fitnesses = [self.fitness(ind, target_melody, target_chord) for ind in population]
            best_fit = max(fitnesses)
            best_sequence = population[fitnesses.index(best_fit)]

            if generation % max(1,(GENERATIONS//10)) == 0:

                melody_pitch = [self.get_melody_pitch(chord) for chord in best_sequence]
                print(f"Generation {generation}: Best Fitness = {best_fit}")
                print("Best sequence melody:", pitch2name(melody_pitch))
                print("Best finger position:", best_sequence)

            candidates = self.tournament_selection(population, fitnesses)
            new_population = []
            candidate_idx = 0
            for _ in range(POPULATION_SIZE-RESERVED):
                candidate_idx %= (len(candidates) - 1)
                parent1 = candidates[candidate_idx]
                parent2 = candidates[candidate_idx + 1]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
                candidate_idx += 1
            population = new_population
            for i in range(RESERVED):
                population.append(candidates[i])


        final_fitnesses = [self.fitness(ind, target_melody,target_chord) for ind in population]
        best_sequence = population[np.argmax(final_fitnesses)]
        return best_sequence

    def run(self):
        """
        如果目标旋律是 list of list，则依次对每个旋律运行遗传算法，
        返回一个包含各个旋律最佳序列的列表；否则返回单个最佳序列。
        """
        results = []
        for i in range(len(self.target_melody_list)):
            target_melody=self.target_melody_list[i]
            target_chord=self.target_chord_list[i]
            print("\nProcessing target melody:", target_melody, target_chord)
            best_sequence = self.run_single(target_melody,target_chord)
            results.append(best_sequence)
        # 如果原始输入为单个旋律，则直接返回该序列，否则返回列表
        if len(self.target_melody_list) == 1:
            return results[0]
        else:
            return results
        

class HandGuitarGA(GuitarGeneticAlgorithm):
    def __init__(self, target_melody, target_chords=None, guitar=None):
        super().__init__(target_melody, target_chords, guitar)

    def get_melody_pitch(self, chord_data):
        """从增强的和弦数据中提取最高音的MIDI pitch"""
        fingering, _ = chord_data  # 解包和弦数据，忽略手指信息
        return super().get_melody_pitch(fingering)

    def cal_finger_comfort(self, finger_positions):
        """计算手指位置的舒适度"""
        if not finger_positions:
            return 0

        active_positions = [pos for pos in finger_positions if pos != -1]
        if not active_positions:
            return 0

        finger_span = max(active_positions) - min(active_positions)
        adjacent_distances = []
        sorted_positions = sorted(active_positions)
        for i in range(len(sorted_positions) - 1):
            adjacent_distances.append(sorted_positions[i + 1] - sorted_positions[i])

        span_penalty = -finger_span * 0.5
        distance_penalty = sum(-abs(d - 2) for d in adjacent_distances) if adjacent_distances else 0

        return span_penalty + distance_penalty

    def cal_finger_movement(self, prev_fingers, curr_fingers):
        """计算两个连续和弦之间的手指移动量"""
        total_movement = 0
        for prev_pos, curr_pos in zip(prev_fingers, curr_fingers):
            if prev_pos != -1 and curr_pos != -1:
                total_movement += abs(curr_pos - prev_pos)
        return -total_movement

    def cal_PC(self, sequence):
        """计算物理舒适度（Physical Comfort）"""
        fingerings, finger_assignments = zip(*sequence)

        PC1 = -sum(sum(1 for fret in chord if fret > 0) for chord in fingerings)
        PC2 = -sum((max(chord) - max(min(chord), 0)) for chord in fingerings)

        total_avg_fret = 0
        for chord in fingerings:
            fretted_notes = [fret for fret in chord if fret > 0]
            avg_fret = sum(fretted_notes) / max(1, len(fretted_notes)) if fretted_notes else 0
            total_avg_fret += avg_fret

        PC3 = total_avg_fret

        PC4 = 0
        for i in range(1, len(fingerings)):
            prev_chord = fingerings[i - 1]
            curr_chord = fingerings[i]
            hand_movement = sum(abs(curr_fret - prev_fret)
                                for curr_fret, prev_fret in zip(curr_chord, prev_chord)
                                if curr_fret > 0 and prev_fret > 0)
            PC4 += hand_movement
        PC4 *= -1

        PC5 = sum(self.cal_finger_comfort(fingers) for fingers in finger_assignments)

        PC6 = 0
        for i in range(1, len(finger_assignments)):
            PC6 += self.cal_finger_movement(finger_assignments[i - 1], finger_assignments[i])

        return (PC1 + PC2 + PC3 + PC4 + PC5 + PC6) / len(sequence)
    
    def cal_NWC(self, sequence, target_melody):
        fingerings, _ = zip(*sequence)
        return super().cal_NWC(fingerings, target_melody)

    def new_cal_NCC(self, play_seq, chord_name):
        fingerings, _ = zip(*play_seq)
        return super().new_cal_NCC(fingerings, chord_name)

    def initialize_population(self, target_melody):
        """初始化种群，现在每个和弦包含按弦位置和手指分配"""
        population = []
        for _ in range(POPULATION_SIZE):
            sequence = []
            for _ in range(len(target_melody)):
                fingering = [random.randint(-1, MAX_FRET) for _ in range(NUM_STRINGS)]
                finger_assignment = [-1] * NUM_STRINGS
                pressed_strings = [i for i, fret in enumerate(fingering) if fret > 0]

                if pressed_strings:
                    available_fingers = list(range(1, NUM_FINGERS + 1))
                    random.shuffle(available_fingers)
                    for i, string_idx in enumerate(pressed_strings):
                        if i < len(available_fingers):
                            finger_assignment[string_idx] = available_fingers[i]

                sequence.append((fingering, finger_assignment))
            population.append(sequence)
        return population

    def mutate(self, sequence):
        """变异操作，同时处理按弦位置和手指分配"""
        mutated_sequence = []
        for fingering, finger_assignment in sequence:
            mutated_fingering = [
                random.randint(-1, MAX_FRET) if random.random() < MUTATION_RATE else fret
                for fret in fingering
            ]

            mutated_fingers = [-1] * NUM_STRINGS
            pressed_strings = [i for i, fret in enumerate(mutated_fingering) if fret > 0]

            if pressed_strings:
                available_fingers = list(range(1, NUM_FINGERS + 1))
                random.shuffle(available_fingers)
                for i, string_idx in enumerate(pressed_strings):
                    if i < len(available_fingers):
                        mutated_fingers[string_idx] = available_fingers[i]

            mutated_sequence.append((mutated_fingering, mutated_fingers))

        return mutated_sequence

    def fitness(self, sequence, target_melody, target_chord):
        """计算适应度"""
        w_PC = 0.0
        w_NWC = 0.0
        w_NCC = 1.0

        PC = self.cal_PC(sequence)
        NWC = self.cal_NWC(sequence, target_melody)
        NCC = self.new_cal_NCC(sequence, target_chord)

        return PC * w_PC + NWC * w_NWC + NCC * w_NCC

    def run_single(self, target_melody, target_chord):
        """运行单个目标旋律的遗传算法"""
        population = self.initialize_population(target_melody)

        for generation in range(GENERATIONS):
            fitnesses = [self.fitness(ind, target_melody, target_chord) for ind in population]
            best_fit = max(fitnesses)
            best_sequence = population[fitnesses.index(best_fit)]

            if generation % max(1, (GENERATIONS // 10)) == 0:
                melody_pitch = [self.get_melody_pitch(chord_data) for chord_data in best_sequence]
                print(f"Generation {generation}: Best Fitness = {best_fit}")
                print("Best sequence melody:", pitch2name(melody_pitch))
                print("Best sequence fingerings:", best_sequence)

            if generation == GENERATIONS - 1:
                result = self.new_cal_NCC(best_sequence, target_chord)
                print('notes not in chord', result * len(best_sequence))

            candidates = self.tournament_selection(population, fitnesses)
            new_population = []
            candidate_idx = 0

            for _ in range(POPULATION_SIZE - RESERVED):
                candidate_idx %= (len(candidates) - 1)
                parent1 = candidates[candidate_idx]
                parent2 = candidates[candidate_idx + 1]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
                candidate_idx += 1

            population = new_population + candidates[:RESERVED]

        final_fitnesses = [self.fitness(ind, target_melody, target_chord) for ind in population]
        best_sequence = population[np.argmax(final_fitnesses)]
        return best_sequence