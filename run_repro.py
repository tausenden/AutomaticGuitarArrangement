from GAutils import Guitar, pitch2name, visualize_guitar_tab, multi_bar_tablature_to_midi
#from GA_repro2_lib import GAreproducing
from ga_reproduction import GAreproducing
import os

def main():
    # Path to the MIDI file
    midi_file_path = 'midis\caihong-4bar.midi'  # Replace with your actual MIDI file path
    
    # Check if file exists
    if not os.path.exists(midi_file_path):
        print(f"Error: MIDI file not found at {midi_file_path}")
        return
    
    print(f"Processing MIDI file: {midi_file_path}")
    
    # Create Guitar instance
    guitar = Guitar()
    
    # Create GA instance with position-based approach
    ga = GAreproducing(
        guitar=guitar,
        midi_file_path=midi_file_path,
        mutation_rate=0.03,
        population_size=100,
        generations=100,
        max_fret=15,
    )
    
    # You can adjust the weights for different fitness components
    ga.weight_PC = 0.8   # Playability
    ga.weight_NWC = 1.5  # Note Weight (higher weight for accurate notes)
    ga.weight_NCC = 1.0  # Notes in Chord

    print("Starting genetic algorithm arrangement...")
    
    # Run the genetic algorithm
    best_tablatures = ga.run()
    
    print("\nArrangement complete!")

    # Save the arrangement as a MIDI file
    if best_tablatures and len(best_tablatures) > 0:
        print("\nSaving arrangement to MIDI file: arrangement_output.mid")
        multi_bar_tablature_to_midi(best_tablatures, guitar, "arrangement_output.mid")

    # Visualize each bar
    print("\nFinal tablature")
    for tab_idx, tab in enumerate(best_tablatures):
        print(f"\nBar {tab_idx+1}:")
        active_positions = [
            chord for pos, chord in enumerate(tab)
            if any(fret != -1 for fret in chord) and ga.bars_data[tab_idx]['original_midi_pitches'][pos]
        ]
        if active_positions:
            visualize_guitar_tab(active_positions)

if __name__ == "__main__":
    main()