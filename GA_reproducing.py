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
from GAlib import GuitarGeneticAlgorithm

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
