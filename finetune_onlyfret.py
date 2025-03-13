import numpy as np
import random
from utlis import Guitar
from utlis import pitch2name
from utlis import visualize_guitar_tab
from GAlib import HandGuitarGA, GuitarGeneticAlgorithm

def main():
    # 创建 Guitar 实例
    guitar = Guitar()

    # 示例1：单个旋律输入
    target_melody = [60, 62, 64, 60, 60, 62, 64, 60]
    target_chord='C'
    print("===== 单旋律情况 =====")
    # Example with custom parameters
    ga = GuitarGeneticAlgorithm(
        target_melody=target_melody,
        target_chords=target_chord,
        guitar=guitar,
        mutation_rate=0.3,
        population_size=1000,
        generations=800,
        reserved_ratio=0.1,
        max_fret=8,
        w_PC=0.0,
        w_NWC=1.0,
        w_NCC=5.0
    )
    best_sequence = ga.run()
    print("\nFinal best sequence:")
    print(best_sequence)
    print("Final best fitness:")
    print(ga.fitness(best_sequence,target_melody,target_chord))
    print("fitness components:")
    print("Melody")
    print(ga.cal_NWC(best_sequence,target_melody))
    print("Chord")
    print(ga.new_cal_NCC(best_sequence,target_chord)*len(best_sequence)) 
    print("PC")
    print(ga.cal_PC(best_sequence))
    melody_pitch = [ga.get_melody_pitch(chord) for chord in best_sequence]
    print("Melody pitches:", pitch2name(melody_pitch))
    print("Target Melody:", pitch2name(target_melody))
    visualize_guitar_tab(best_sequence)
    # 示例2：list of list 输入
    target_melody_s = [
        [60, 0, 62, 0, 64, 0, 60, 0],
        [64, 0, 65, 0, 67, 0, 0, 0],
        [64, 0, 55, 0, 60, 0, -1, -1]
    ]
    target_chords=['C','G','C']

    print("\n===== 多旋律情况 =====")
    ga_m = GuitarGeneticAlgorithm(
        target_melody=target_melody_s,
        target_chords=target_chords,
        guitar=guitar,
        mutation_rate=0.4,
        population_size=1000,
        generations=1200,
        max_fret=8,
        w_PC=0.0,
        w_NWC=1.0,
        w_NCC=0.0
    )
    # best_sequences = ga_m.run()
    # for idx, seq in enumerate(best_sequences):
    #     print(f"\nFinal best fingering for melody {idx}:")
    #     print(seq)
    #     melody_pitch = [ga_m.get_melody_pitch(chord) for chord in seq]
    #     print("Melody pitches:", pitch2name(melody_pitch))
    #     print("Target Melody:", pitch2name(target_melody_s[idx]))

if __name__ == "__main__":
    main()
