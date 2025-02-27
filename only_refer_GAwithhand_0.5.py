import numpy as np
import random
from utlis import Guitar
from utlis import pitch2name
from utlis import visualize_guitar_tab

# 常量定义
MUTATION_RATE = 0.3
POPULATION_SIZE = 1000
GENERATIONS = 100
TOURNAMENT_SIZE = 1000
RESERVED = POPULATION_SIZE//10
NUM_STRINGS = 6
MAX_FRET = 8
NUM_FINGERS = 4  # 不包括大拇指

class EnhancedGuitarGeneticAlgorithm:
    def __init__(self, target_melody, target_chords=None, guitar=None):
        if guitar is None:
            assert('no guitar wrong')
            
        self.guitar = guitar
        
        if isinstance(target_melody, list) and target_melody and isinstance(target_melody[0], list):
            self.target_melody_list = target_melody
        else:
            self.target_melody_list = [target_melody]

        if isinstance(target_chords, list) and target_chords:
            self.target_chord_list = target_chords
        else:
            self.target_chord_list = [target_chords]

    def get_melody_pitch(self, chord_data):
        """从增强的和弦数据中提取最高音的MIDI pitch"""
        fingering, _ = chord_data  # 解包和弦数据，忽略手指信息
        fingering_dict = {i+1: fret for i, fret in enumerate(fingering)}
        midi_notes = self.guitar.get_chord_midi(fingering_dict)
        return max(midi_notes) if midi_notes else None

    def cal_finger_comfort(self, finger_positions):
        """计算手指位置的舒适度"""
        if not finger_positions:
            return 0
            
        # 过滤掉未使用的手指(-1)
        active_positions = [pos for pos in finger_positions if pos != -1]
        if not active_positions:
            return 0
            
        # 计算手指跨度
        finger_span = max(active_positions) - min(active_positions)
        
        # 计算相邻手指之间的距离
        adjacent_distances = []
        sorted_positions = sorted(active_positions)
        for i in range(len(sorted_positions)-1):
            adjacent_distances.append(sorted_positions[i+1] - sorted_positions[i])
        
        # 手指跨度惩罚（跨度越大越不舒服）
        span_penalty = -finger_span * 0.5
        
        # 相邻手指距离惩罚（距离过大或过小都不舒服）
        distance_penalty = sum(-abs(d-2) for d in adjacent_distances) if adjacent_distances else 0
        
        return span_penalty + distance_penalty

    def cal_finger_movement(self, prev_fingers, curr_fingers):
        """计算两个连续和弦之间的手指移动量"""
        total_movement = 0
        for prev_pos, curr_pos in zip(prev_fingers, curr_fingers):
            if prev_pos != -1 and curr_pos != -1:
                total_movement += abs(curr_pos - prev_pos)
        return -total_movement  # 负值因为我们想要最小化移动
    
    def cal_PC(self, sequence):
        """计算物理舒适度（Physical Comfort）"""
        fingerings, finger_assignments = zip(*sequence)  # 解包序列中的和弦数据
        
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

        # 添加手指舒适度评分
        PC5 = sum(self.cal_finger_comfort(fingers) for fingers in finger_assignments)
        
        # 添加手指移动评分
        PC6 = 0
        for i in range(1, len(finger_assignments)):
            PC6 += self.cal_finger_movement(finger_assignments[i-1], finger_assignments[i])

        return (PC1 + PC2 + PC3 + PC4 + PC5 + PC6) / len(sequence)

    def cal_NWC(self, sequence, target_melody):
        """计算音符权重一致性（Note Weight Consistency）"""
        fingerings, _ = zip(*sequence)  # 解包序列，忽略手指信息
        
        total_error = 0
        valid_notes = 0

        for i in range(min(len(fingerings), len(target_melody))):
            chord_dict = {i+1: fret for i, fret in enumerate(fingerings[i])}
            midi_notes = self.guitar.get_chord_midi(chord_dict)
            
            if not midi_notes:
                total_error += 100
                continue
                
            target_pitch = target_melody[i]
            # closest_pitch_error = min(abs(note - target_pitch) for note in midi_notes)
            
            # pitch_error = closest_pitch_error
            # if max(midi_notes) != target_pitch:
            #     pitch_error += 10
            pitch_error=abs(max(midi_notes)-target_pitch)

            total_error += pitch_error
            valid_notes += 1
        
        return -(total_error) / len(sequence)

    def cal_NCC(self, sequence, chord_name):
        """计算和弦一致性（Note Chord Consistency）"""
        if not chord_name:
            return 0
            
        fingerings, _ = zip(*sequence)  # 解包序列，忽略手指信息
        
        tot_err = 0
        chord_dict = self.guitar.chords[chord_name][0]
        chord_six = [chord_dict[i+1] for i in range(6)]
        
        for fingering in fingerings:
            for i in range(6):
                tot_err += abs(fingering[i] - chord_six[i])
                
        return -(tot_err) / len(sequence)
    
    def new_cal_NCC(self, play_seq, chord_name): # how many notes not in the triad of chord

        seq_fret, _ = zip(*play_seq)
        #print('play_seq:',seq_fret)
        tot_err=0
        target_chord_note=self.guitar.chords4NCC[chord_name]

        if not chord_name:
            return 0
        for i in range(len(seq_fret)):
            chord_dict = {i+1: fret for i, fret in enumerate(seq_fret[i])}
            midi_notes = self.guitar.get_chord_midi(chord_dict)
            for note in midi_notes:
                note%=12
                if note!=-1 and note not in target_chord_note:
                    tot_err+=1
                    
        return -(tot_err)/len(play_seq)
    
    def fitness(self, sequence, target_melody, target_chord):
        """计算适应度"""
        w_PC = 0.0
        w_NWC = 0.0
        w_NCC = 1.0

        PC = self.cal_PC(sequence)
        NWC = self.cal_NWC(sequence, target_melody)
        # NCC = self.cal_NCC(sequence, target_chord)
        NCC = self.new_cal_NCC(sequence, target_chord)

        return PC * w_PC + NWC * w_NWC + NCC * w_NCC

    def initialize_population(self, target_melody):
        """初始化种群，现在每个和弦包含按弦位置和手指分配"""
        population = []
        for _ in range(POPULATION_SIZE):
            sequence = []
            for _ in range(len(target_melody)):
                # 生成按弦位置
                fingering = [random.randint(-1, MAX_FRET) for _ in range(NUM_STRINGS)]
                
                # 生成手指分配 (-1表示不使用该弦)
                finger_assignment = [-1] * NUM_STRINGS
                pressed_strings = [i for i, fret in enumerate(fingering) if fret > 0]
                
                if pressed_strings:
                    available_fingers = list(range(1, NUM_FINGERS + 1))  # 1-4表示食指到小指
                    random.shuffle(available_fingers)
                    for i, string_idx in enumerate(pressed_strings):
                        if i < len(available_fingers):
                            finger_assignment[string_idx] = available_fingers[i]
                
                sequence.append((fingering, finger_assignment))
            population.append(sequence)
        return population
    
    def tournament_selection(self, population, fitnesses):
        """锦标赛选择：随机选取部分个体，并根据适应度排序"""
        candidates_idx = random.sample(range(len(population)), TOURNAMENT_SIZE)
        candidates_idx.sort(key=lambda idx: fitnesses[idx], reverse=True)
        sorted_candidates = [population[idx] for idx in candidates_idx]
        return sorted_candidates

    def crossover(self, parent1, parent2):
        """交叉操作，同时处理按弦位置和手指分配"""
        if len(parent1) < 2:
            return parent1.copy()
            
        points = sorted(random.sample(range(len(parent1)), 2))
        child = (
            parent1[:points[0]] +
            parent2[points[0]:points[1]] +
            parent1[points[1]:]
        )
        return child

    def mutate(self, sequence):
        """变异操作，同时处理按弦位置和手指分配"""
        mutated_sequence = []
        for fingering, finger_assignment in sequence:
            # 变异按弦位置
            mutated_fingering = [
                random.randint(-1, MAX_FRET) if random.random() < MUTATION_RATE else fret
                for fret in fingering
            ]
            
            # 相应地更新手指分配
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

    def run_single(self, target_melody, target_chord):
        """运行单个目标旋律的遗传算法"""
        population = self.initialize_population(target_melody)

        for generation in range(GENERATIONS):
            fitnesses = [self.fitness(ind, target_melody, target_chord) for ind in population]
            best_fit = max(fitnesses)
            best_sequence = population[fitnesses.index(best_fit)]

            if generation % max(1, (GENERATIONS//10)) == 0:
                melody_pitch = [self.get_melody_pitch(chord_data) for chord_data in best_sequence]
                print(f"Generation {generation}: Best Fitness = {best_fit}")
                print("Best sequence melody:", pitch2name(melody_pitch))
                print("Best sequence fingerings:", best_sequence)

            if generation == GENERATIONS-1:
                result=self.new_cal_NCC(best_sequence,target_chord)
                print('notes not in chord',result*len(best_sequence))

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

    def run(self):
        """运行算法，处理单个或多个旋律"""
        results = []
        for i in range(len(self.target_melody_list)):
            target_melody = self.target_melody_list[i]
            target_chord = self.target_chord_list[i]
            print("\nProcessing target melody:", target_melody, target_chord)
            best_sequence = self.run_single(target_melody, target_chord)
            results.append(best_sequence)
            
        return results[0] if len(self.target_melody_list) == 1 else results

def main():
    guitar = Guitar()
    
    # 测试单个旋律
    target_melody = [60, 0, 62, 0, 64, 0, 60, 0, 60, 0, 62, 0, 64, 0, 60, 0]
    target_chord = 'C'
    print("===== 单旋律测试 =====")
    ga = EnhancedGuitarGeneticAlgorithm(target_melody, target_chord, guitar)
    best_sequence = ga.run()
    print("\nFinal best sequence:")
    print("Pressing and Fingering assignments:", best_sequence)
    melody_pitch = [ga.get_melody_pitch(chord_data) for chord_data in best_sequence]
    print("Melody pitches:", pitch2name(melody_pitch))
    print("Target Melody:", pitch2name(target_melody))
    visualize_guitar_tab(best_sequence)

if __name__ == "__main__":
    main()