from debug_newGAlib import GAimproved
from GAutils import Guitar, set_random
from ga_export import export_ga_results
from remi_z import MultiTrack
set_random(42)

ga_config={
    'guitar':Guitar(),
    'weight_PC':1.0,
    'weight_NWC':1.0,
    'weight_NCC':2.0,
    'weight_RP':1.0,
    'mutation_rate':0.03,
    'crossover_rate':0.6,
    'population_size':300,
    'generations':100,
    'max_fret':15,
    'tournament_k': 5,
    'midi_file_path':'caihong_clip/caihong_clip_4_12.mid',
}
mt=MultiTrack.from_midi(ga_config['midi_file_path'])
tempo=mt.tempos[0]
ga = GAimproved(**ga_config)
ga_candidate=ga.run()
output_files = export_ga_results(
            ga_tab_seq=ga_candidate,
            song_name='caihong_clip_4_12_debug',
            tempo=tempo,
            output_dir='debug_folder1',
            sf2_path='resources/Tyros Nylon.sf2'
        )