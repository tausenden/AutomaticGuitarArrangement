from GAutils import midi_process
import numpy as np
import random
from GAutils import Guitar
from GAutils import pitch2name
from GAutils import visualize_guitar_tab
from GA_workonRP_lib import RhythmGA
midi_file_path = "midis\once_guitar.mid"
guitar = Guitar()
t_melody,t_chord= midi_process(midi_file_path)
ga_tool=RhythmGA(target_melody=t_melody, target_chords=t_chord, guitar=guitar)
def main1():
    ga = RhythmGA(
        target_melody=t_melody,
        target_chords=t_chord,
        guitar=guitar,
        mutation_rate=0.3,
        population_size=400,
        generations=100,
        reserved_ratio=0.05,
        max_fret=19,
        midi_file_path=midi_file_path,
        w_PC=1.0,
        w_NWC=1.0,
        w_NCC=1.0,
        w_RP=1.0
    )
    best_sequences=ga.run()
    for idx, seq in enumerate(best_sequences):
        print(f"\nFinal best fingering for melody {idx}:")
        melody_pitch = [ga.get_melody_pitch(chord) for chord in seq]
        print("Melody pitches:", pitch2name(melody_pitch))
        print("Target Melody:", pitch2name(t_melody[idx]))
        visualize_guitar_tab(seq)
        print(f'PC: {ga.cal_PC(seq, verbose=0)*len(t_melody[idx])}')
        print(f'NWC: {ga.cal_NWC(seq,t_melody[idx])*len(t_melody[idx])}')
        print(f'NCC: {ga.cal_NCC(seq,t_melody[idx],t_chord[idx],verbose=0)*len(t_melody[idx])}')

if __name__ == "__main__":
    main1()

