from GAutils import Guitar, pitch2name, visualize_guitar_tab, visualize_guitar_tab_withpos
from GA_repro2_lib import GAreproducing
import os

def main():
    # Path to the MIDI file
    midi_file_path = 'testremiz\misc\caihong-4bar.midi'  # Replace with your actual MIDI file path
    
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
        population_size=500,
        generations=300,
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
    show_position=1

    if not show_position:
        if best_tablatures and len(best_tablatures) > 0:
            print("\nFinal tablature")
            
            # Extract active positions for visualization
            active_positions = []
            for tab_idx in range(len(best_tablatures)):
                for pos, chord in enumerate(best_tablatures[tab_idx]):
                    if any(fret != -1 for fret in chord) and ga.bars_data[0]['positions'][pos] > 0:
                        active_positions.append(chord)
                
            if active_positions:
                visualize_guitar_tab(active_positions)

    if show_position:
        if best_tablatures and len(best_tablatures) > 0:
            print("\nFinal tablature")
            
            # Process each bar separately
            for tab_idx in range(len(best_tablatures)):
                print(f"\nBar {tab_idx+1}:")
                
                # Collect the active positions (with their actual position numbers)
                active_positions = []
                position_map = {}  # To map sequence indices to actual MIDI positions
                
                for pos, chord in enumerate(best_tablatures[tab_idx]):
                    # Only include positions that have notes and match melody notes
                    if any(fret != -1 for fret in chord) and ga.bars_data[tab_idx]['positions'][pos] > 0:
                        active_positions.append(chord)
                        position_map[len(active_positions) - 1] = pos  # Map the index in active_positions to original position
                
                if active_positions:
                    # Visualize with position mapping
                    visualize_guitar_tab_withpos(active_positions, position_map)

    # You can save or process the tablatures further here
    # For example, you could convert them to MIDI or display them as sheet music

if __name__ == "__main__":
    main()