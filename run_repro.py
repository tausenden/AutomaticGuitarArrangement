from GAutils import Guitar, pitch2name, visualize_guitar_tab, multi_bar_tablature_to_midi
#from GA_repro2_lib import GAreproducing
from ga_reproduction import GAreproducing
import os
import time

def main():
    # Configuration
    input_folder = 'test_arr_midis'  # Fixed input folder
    output_folder = 'test_arranged_midis'     # Fixed output folder
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all MIDI files
    midi_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mid', '.midi'))]
    
    for filename in midi_files:
        midi_file_path = os.path.join(input_folder, filename)
        
        # Get output filename (keep same name, normalize .midi to .mid)
        if filename.lower().endswith('.midi'):
            output_name = filename[:-5] + '.mid'
        else:
            output_name = filename
        output_midi_path = os.path.join(output_folder, output_name)
        
        # GA parameters
        ga_config = {
            'mutation_rate': 0.03,
            'population_size': 400,
            'generations': 120,
            'max_fret': 15,
        }
        
        # Fitness weights
        fitness_weights = {
            'PC': 1.0,    # Playability
            'NWC': 1.5,   # Note Weight (higher weight for accurate notes)
            'NCC': 1.0,   # Notes in Chord
        }
        
        # MIDI export settings
        midi_settings = {
            'tempo': 140,
            'time_signature': (4, 4),
            'velocity': 100,
            'duration': 12,  # 16th note duration
            'inst_id': 25,   # Acoustic Guitar
        }
        
        print(f"Processing MIDI file: {midi_file_path}")
        print(f"Output will be saved to: {output_midi_path}")
        
        # Create Guitar instance
        guitar = Guitar()
        
        # Create GA instance with position-based approach
        print("\nInitializing Genetic Algorithm...")
        ga = GAreproducing(
            guitar=guitar,
            midi_file_path=midi_file_path,
            **ga_config
        )
        
        # Set fitness weights
        ga.weight_PC = fitness_weights['PC']
        ga.weight_NWC = fitness_weights['NWC']
        ga.weight_NCC = fitness_weights['NCC']
        
        print(f"GA Configuration:")
        print(f"  Population: {ga_config['population_size']}")
        print(f"  Generations: {ga_config['generations']}")
        print(f"  Mutation Rate: {ga_config['mutation_rate']}")
        print(f"  Max Fret: {ga_config['max_fret']}")
        print(f"Fitness Weights:")
        print(f"  Playability (PC): {fitness_weights['PC']}")
        print(f"  Note Weight (NWC): {fitness_weights['NWC']}")
        print(f"  Notes in Chord (NCC): {fitness_weights['NCC']}")

        print("\nStarting genetic algorithm arrangement...")
        start_time = time.time()
        
        # Run the genetic algorithm
        best_tablatures = ga.run()
        
        end_time = time.time()
        print(f"\nArrangement complete! Time taken: {end_time - start_time:.2f} seconds")

        # Save the arrangement as a MIDI file
        print(f"\nSaving arrangement to MIDI file: {output_midi_path}")
        
        # 使用新的GATabSeq转换方法
        mt = best_tablatures.convert_to_multitrack(
            guitar, 
            tempo=midi_settings['tempo'],
            time_signature=midi_settings['time_signature']
        )

        mt.to_midi(output_midi_path)
        print(f"✓ MIDI file saved successfully: {output_midi_path}")

        # Visualize each bar
        print("\nFinal tablature visualization:")
        print(str(best_tablatures))  # 使用新的字符串表示
        
        
        print(f"\nProcess completed!")
        print(f"Input: {midi_file_path}")
        print(f"Output: {output_midi_path}")

if __name__ == "__main__":
    main()