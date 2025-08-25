from GAutils import Guitar, set_random
from GA_improved import GAimproved
from ga_export import export_ga_results
import os

def main():
    set_random(42)
    
    # Configuration
    input_folder = 'caihong_clip'
    output_base_folder = 'arranged_caihong_clip_GA_improved'
    
    # GA parameters
    ga_config = {
        'mutation_rate': 0.03,
        'crossover_rate': 0.6,
        'population_size': 300,
        'generations': 100,
        'num_strings': 6,
        'max_fret': 15,
        'weight_PC': 1.0,
        'weight_NWC': 1.5,
        'weight_NCC': 1.0,
        'weight_RP': 1.0,
        'tournament_k': 5,
    }
    
    # Export settings
    export_settings = {
        'tempo': 80,
        'sf2_path': 'resources/Tyros Nylon.sf2'
    }
    
    os.makedirs(output_base_folder, exist_ok=True)
    
    midi_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mid', '.midi'))]
    
    for filename in midi_files:
        midi_file_path = os.path.join(input_folder, filename)
        song_name = os.path.splitext(filename)[0]
        song_output_folder = os.path.join(output_base_folder, song_name)
        
        print(f"Processing: {filename}")
        
        guitar = Guitar()
        
        ga = GAimproved(
            guitar=guitar,
            midi_file_path=midi_file_path,
            **ga_config
        )
        
        ga_tab_seq = ga.run()
        
        output_files = export_ga_results(
            ga_tab_seq=ga_tab_seq,
            song_name=song_name,
            tempo=export_settings['tempo'],
            output_dir=song_output_folder,
            sf2_path=export_settings['sf2_path']
        )

if __name__ == "__main__":
    main()