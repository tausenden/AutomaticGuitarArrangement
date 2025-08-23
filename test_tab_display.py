from GAutils import GATab, GATabSeq, Guitar, set_random
from ga_reproduction import GAreproducing
import os

def main():
    set_random(42)
    
    # Single MIDI file to test
    midi_file_path = 'caihong_clip/caihong_clip_4_12.mid'  # Change this to your test file
    song_name = 'test_song'
    
    # Simple GA parameters for quick testing
    ga_config = {
        'mutation_rate': 0.03,
        'population_size': 50,
        'generations': 20,
        'max_fret': 15,
    }
    
    print(f"Processing: {midi_file_path}")
    
    guitar = Guitar()
    
    ga = GAreproducing(
        guitar=guitar,
        midi_file_path=midi_file_path,
        **ga_config
    )
    
    ga_tab_seq = ga.run()
    
    # Save the tab for inspection
    ga_tab_seq.save_to_file(f"{song_name}.tab")
    
    print("Tab sequence created. You can inspect 'ga_tab_seq' in debug console.")
    return ga_tab_seq

if __name__ == "__main__":
    tab_seq = main()  # Run and store result for debug console inspection
