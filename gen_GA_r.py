from core import Guitar, set_random, GAreproducing, export_ga_results
from remi_z import MultiTrack
import os
import time
import json

def main():
    set_random(42)
    
    # Configuration
    input_folder = 'song_midis'
    output_base_folder = 'arranged_songs_GA_r'
    
    # GA parameters
    ga_config = {
        'mutation_rate': 0.03,
        'crossover_rate': 0.6,
        'population_size': 500,
        'generations': 100,
        'num_strings': 6,
        'max_fret': 15,
        'weight_PC': 1.0,
        'weight_NWC': 1.5,
        'weight_NCC': 1.0,
        'tournament_k': 5,
        'resolution': 16
    }
    
    # Export settings
    export_settings = {
        'sf2_path': 'resources/Tyros Nylon.sf2'
    }
    
    os.makedirs(output_base_folder, exist_ok=True)
    
    midi_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mid', '.midi'))]
    
    for filename in midi_files:
        midi_file_path = os.path.join(input_folder, filename)
        song_name = os.path.splitext(filename)[0]
        song_output_folder = os.path.join(output_base_folder, song_name)

        # Skip if arrangement already exists
        if os.path.isdir(song_output_folder):
            print(f"Skipping {song_name}: output folder already exists at {song_output_folder}")
            continue

        # Process with simple error handling inline
        try:
            print(f"Processing: {filename}")
            start_time = time.time()

            guitar = Guitar()
            ga = GAreproducing(
                guitar=guitar,
                midi_file_path=midi_file_path,
                **ga_config
            )

            ga_tab_seq = ga.run()
            get_tempo = MultiTrack.from_midi(midi_file_path).tempos[0]

            end_time = time.time()
            processing_time = end_time - start_time

            output_files = export_ga_results(
                ga_tab_seq=ga_tab_seq,
                song_name=song_name,
                tempo=get_tempo,
                output_dir=song_output_folder,
                sf2_path=export_settings['sf2_path'],
                resolution=ga_config['resolution']
            )

            ga_stats_file = ga.export_statistics(song_output_folder, song_name)

            with open(ga_stats_file, 'r') as f:
                ga_data = json.load(f)

            ga_data.update({
                'processing_time_seconds': round(processing_time, 2),
            })

            with open(ga_stats_file, 'w') as f:
                json.dump(ga_data, f, indent=2)

            print(f"Generated arrangement for {song_name} in {processing_time:.2f} seconds")
        except Exception as e:
            print(f"Error processing {song_name}: {e}")
            continue

if __name__ == "__main__":
    main()