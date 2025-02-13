import numpy as np
import random
from utlis import Guitar

# Define constants
MUTATION_RATE = 0.2
POPULATION_SIZE = 300
GENERATIONS = 300
TOURNAMENT_SIZE = 300
SEQUENCE_LENGTH = 4  # 序列中的和弦数量
NUM_STRINGS = 6  # 吉他弦数
MAX_FRET = 16 # 最大品格数

class GuitarGeneticAlgorithm:
    def __init__(self, target_melody, guitar):
        self.target_melody = target_melody  # 目标MIDI音高序列
        self.guitar = guitar  # Guitar类实例
        #self.target_chord=target_chord
        
    def get_melody_pitch(self, chord_fingering):
        """从和弦指法中提取最高音的MIDI pitch"""
        # 将列表形式的指法转换为字典形式
        fingering_dict = {i+1: fret for i, fret in enumerate(chord_fingering)}
        # 获取所有音符的MIDI pitch
        midi_notes = self.guitar.get_chord_midi(fingering_dict)
        # 返回最高音的MIDI pitch（如果所有弦都不弹则返回None）
        return max(midi_notes) if midi_notes else None

    # def fitness(self, sequence):
    #     """计算序列与目标旋律的相似度"""
    #     total_error = 0
    #     valid_notes = 0
        
    #     # 确保序列长度与目标旋律长度相匹配
    #     for i in range(min(len(sequence), len(self.target_melody))):
    #         melody_pitch = self.get_melody_pitch(sequence[i])
    #         if melody_pitch is not None:
    #             # 计算与目标音高的差距（音程差的绝对值）
    #             pitch_error = abs(melody_pitch - self.target_melody[i])
    #             total_error += pitch_error
    #             valid_notes += 1
        
    #     if valid_notes == 0:
    #         return 0  # 如果没有有效音符，返回最低适应度
            
    #     # 返回负的平均误差（误差越小，适应度越高）
    #     average_error = total_error / valid_notes
    #     return -average_error

    def cal_PC(self, sequence):
        PC1 = -sum(sum(1 for fret in chord if fret > 0) for chord in sequence)  # string press in the same time
        PC2 = -sum((max(chord) - max(min(chord), 0)) for chord in sequence)  # width of the press
        PC3 = sum(sum(fret for fret in chord if fret > 0) / sum(1 for fret in chord if fret > 0) for chord in sequence) / len(sequence)  # average fret press
        PC4 = 0 # not really hand movement, just press diff, this should be in the hand part
        for i in range(1, len(sequence)):
            prev_chord = sequence[i - 1]
            curr_chord = sequence[i]
            hand_movement = sum(abs(curr_fret - prev_fret) for curr_fret, prev_fret in zip(curr_chord, prev_chord) if curr_fret > 0 and prev_fret > 0)
            PC4 += hand_movement
            # 整个手的移动
        PC4 *= -1 
        return PC1 + PC2 + PC3 + PC4
    
    def cal_NWC(self, sequence):

        total_error = 0
        valid_notes = 0

        for i in range(min(len(sequence), len(self.target_melody))):
            chord_dict = {i+1: fret for i, fret in enumerate(sequence[i])}
            midi_notes = self.guitar.get_chord_midi(chord_dict)
            
            if not midi_notes:  # 如果和弦没有音符
                total_error += 100  # 惩罚空和弦
                continue
                
            # 找到最接近目标音高的音符
            # position 预测，结合guitar类
            target_pitch = self.target_melody[i]
            if target_pitch == -1:
                for note in midi_notes:
                    if note != -1:
                        total_error += 100

            closest_pitch_error = min(abs(note - target_pitch) for note in midi_notes)*10
            
            # 计算误差并加权
            pitch_error = closest_pitch_error
            if max(midi_notes) != target_pitch:  # 如果最高音不是目标音高
                pitch_error += 5  # 额外惩罚
                
            total_error += pitch_error
            valid_notes += 1

    def cal_NCC(self, sequence): #chord in class, 完全等价吗?
        for chord in sequence:
            chord_dict = {i+1: fret for i, fret in enumerate(sequence)}
        pass

    def fitness(self, sequence):
        """改进的适应度函数"""
        total_error = 0
        valid_notes = 0
        
        for i in range(min(len(sequence), len(self.target_melody))):
            chord_dict = {i+1: fret for i, fret in enumerate(sequence[i])}
            midi_notes = self.guitar.get_chord_midi(chord_dict)
            
            if not midi_notes:  # 如果和弦没有音符
                total_error += 100  # 惩罚空和弦
                continue
                
            # 找到最接近目标音高的音符
            # position 预测，结合guitar类
            target_pitch = self.target_melody[i]
            closest_pitch_error = min(abs(note - target_pitch) for note in midi_notes)*10
            
            # 计算误差并加权
            pitch_error = closest_pitch_error
            if max(midi_notes) != target_pitch:  # 如果最高音不是目标音高
                pitch_error += 5  # 额外惩罚
                
            total_error += pitch_error
            valid_notes += 1
        
        # 添加对和弦复杂度的惩罚
        # 实现方法再换一下
        #complexity_penalty = sum(sum(1 for fret in chord if fret > 0) for chord in sequence)

        NWC= -(total_error) / len(sequence)
        PC1=-sum(sum(1 for fret in chord if fret > 0) for chord in sequence) #同下，特化横按
        PC2=-sum((max(chord) - max(min(chord),0)) for chord in sequence) #可以尝试针对性大幅度增加对超过某个范围的按法的惩罚
        #NCC
        #PC3:高把位惩罚
        #PC4：movement
        PC=PC1+PC2/5
        # 计算最终适应度（负值，越接近0越好）
        fitness_score = (NWC*50+PC)/100
        return fitness_score

    def initialize_population(self):
        # 随机生成初始种群质量太差了
        population = []
        for _ in range(POPULATION_SIZE):
            sequence = []
            for _ in range(len(self.target_melody)):
                # 生成一个和弦指法（六元组）
                chord = [random.randint(-1, MAX_FRET) for _ in range(NUM_STRINGS)]
                sequence.append(chord)
            population.append(sequence)
        return population

    def tournament_selection(self, population, fitnesses):
        """改进的锦标赛选择"""
        # 增加竞标赛大小，使选择压力更大
        candidates = random.sample(range(len(population)), TOURNAMENT_SIZE)
        
        # 根据适应度排序候选者
        candidates.sort(key=lambda x: fitnesses[x], reverse=True)
        sorted_population = [population[candidate] for candidate in candidates]
        
        return sorted_population

    # 已有的两种方法都收敛到局部最小值，还需要更强的变异
    def crossover(self, parent1, parent2):
        points = sorted(random.sample(range(len(parent1)), 2))
        child = (parent1[:points[0]] + 
                parent2[points[0]:points[1]] + 
                parent1[points[1]:])
        return child

    def mutate(self, sequence):
        """变异操作"""
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

    def run(self):
        """运行遗传算法"""
        population = self.initialize_population()
        print(population[0:3])
        
        for generation in range(GENERATIONS):
            # 计算当前种群的适应度
            fitnesses = [self.fitness(ind) for ind in population]
            
            # 记录最佳适应度
            best_fit = max(fitnesses)
            best_sequence = population[fitnesses.index(best_fit)]
            
            if generation % 10 == 0:
                print(f"Generation {generation}: Best Fitness = {best_fit}")
                print("Best sequence melody:", [self.get_melody_pitch(chord) for chord in best_sequence])
            
            # 生成新种群
            new_population = []
            candi=self.tournament_selection(population, fitnesses)
            indselect=0
            for _ in range(POPULATION_SIZE):
                indselect = indselect%3
                parent1 = candi[indselect]
                parent2 = candi[indselect+1]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
                indselect+=1
            
            population = new_population
        
        # 返回最终的最佳序列
        final_fitnesses = [self.fitness(ind) for ind in population]
        best_sequence = population[np.argmax(final_fitnesses)]
        return best_sequence

# 使用示例
def main():
    # 创建Guitar实例
    guitar = Guitar()
    
    # 目标旋律（MIDI音高序列）
    target_melody = [60, 62, 64, 60, 60, 62, 64, 60]  # 示例旋律
    # target_melody=[[60,'-',62,'-',64,'-',60,'-'],[64,'-',65,'-',67,'-','-','-']]
    target_melody=[[60,0,62,0,64,0,60,0],[64,0,65,0,67,0,0,0],[64, 0, 55, 0, 60, 0, -1, -1]]
    target_chord=['C','G','C'] #改成半小节
    
    # 创建并运行遗传算法
    ga = GuitarGeneticAlgorithm(target_melody, guitar)

    best_sequence = ga.run()
    
    print("\nFinal best sequence:")
    print(best_sequence)
    melody_pitch=[ga.get_melody_pitch(chord) for chord in best_sequence]
    print("Melody pitches:", melody_pitch)
    print("Target Melody:", target_melody)
    print("ACC:", sum(1 for x, y in zip(melody_pitch, target_melody) if x == y) / len(target_melody))

if __name__ == "__main__":
    main()