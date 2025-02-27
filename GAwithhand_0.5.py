import numpy as np
import random
from utlis import Guitar
from utlis import pitch2name
from utlis import visualize_guitar_tab
from GAlib import HandGuitarGA

# 常量定义
MUTATION_RATE = 0.3
POPULATION_SIZE = 1000
GENERATIONS = 100
TOURNAMENT_SIZE = 1000
RESERVED = POPULATION_SIZE//10
NUM_STRINGS = 6
MAX_FRET = 8
NUM_FINGERS = 4  # 不包括大拇指

def main():
    guitar = Guitar()
    
    # 测试单个旋律
    target_melody = [60, 0, 62, 0, 64, 0, 60, 0, 60, 0, 62, 0, 64, 0, 60, 0]
    target_chord = 'C'
    print("===== 单旋律测试 =====")
    ga = HandGuitarGA(target_melody, target_chord, guitar)
    best_sequence = ga.run()
    print("\nFinal best sequence:")
    print("Pressing and Fingering assignments:", best_sequence)
    melody_pitch = [ga.get_melody_pitch(chord_data) for chord_data in best_sequence]
    print("Melody pitches:", pitch2name(melody_pitch))
    print("Target Melody:", pitch2name(target_melody))
    visualize_guitar_tab(best_sequence)

if __name__ == "__main__":
    main()