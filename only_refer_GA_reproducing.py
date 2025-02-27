import numpy as np
import random
from utlis import Guitar
from utlis import pitch2name
# 定义常量
MUTATION_RATE = 0.3
POPULATION_SIZE = 1000
GENERATIONS = 800
TOURNAMENT_SIZE = 1000
RESERVED= POPULATION_SIZE//5
NUM_STRINGS = 6  # 吉他弦数
MAX_FRET = 8    # 最大品格数

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

# 使用示例
def main():
    # 创建 Guitar 实例
    guitar = Guitar()

    # 示例1：单个旋律输入
    target_melody = [60, 62, 64, 60, 60, 62, 64, 60]
    target_chord='C'
    print("===== 单旋律情况 =====")
    ga_single = GuitarGeneticAlgorithm(target_melody,target_chord, guitar)
    best_sequence = ga_single.run()
    print("\nFinal best sequence:")
    print(best_sequence)
    melody_pitch = [ga_single.get_melody_pitch(chord) for chord in best_sequence]
    print("Melody pitches:", pitch2name(melody_pitch))
    print("Target Melody:", pitch2name(target_melody))

    # 示例2：list of list 输入
    target_melody_s = [
        [60, 0, 62, 0, 64, 0, 60, 0],
        [64, 0, 65, 0, 67, 0, 0, 0],
        [64, 0, 55, 0, 60, 0, -1, -1]
    ]
    target_chords=['C','G','C']

    # print("\n===== 多旋律情况 =====")
    # ga_multiple = GuitarGeneticAlgorithm(target_melody_s,target_chords, guitar)
    # best_sequences = ga_multiple.run()
    # for idx, seq in enumerate(best_sequences):
    #     print(f"\nFinal best sequence for melody {idx}:")
    #     print(seq)
    #     melody_pitch = [ga_multiple.get_melody_pitch(chord) for chord in seq]
    #     print("Melody pitches:", melody_pitch)
    #     print("Target Melody:", target_melody_s[idx])

if __name__ == "__main__":
    main()
