from remi_z import MultiTrack as mt
import os
import glob
from pathlib import Path

def process_midi_folder(input_folder, output_folder, bars_per_clip=8):
    """
    Process all MIDI files in input_folder and cut them into 8-bar clips
    
    Args:
        input_folder (str): Path to folder containing MIDI files
        output_folder (str): Path to output folder for clips
        bars_per_clip (int): Number of bars per clip (default: 8)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all MIDI files in input folder
    midi_extensions = ['*.mid', '*.midi']
    midi_files = []
    for ext in midi_extensions:
        midi_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    if not midi_files:
        print(f"No MIDI files found in {input_folder}")
        return
    
    print(f"Found {len(midi_files)} MIDI files to process")
    
    for midi_file in midi_files:
        try:
            # Get song name without extension
            song_name = Path(midi_file).stem
            print(f"\nProcessing: {song_name}")
            
            # Load MIDI file
            multitrack = mt.from_midi(midi_file)
            total_bars = len(multitrack)
            print(f"  Total bars: {total_bars}")
            
            # Create output subfolder for this song
            song_output_dir = os.path.join(output_folder, f"{song_name}_clips")
            os.makedirs(song_output_dir, exist_ok=True)
            
            # Calculate number of clips possible
            num_clips = total_bars // bars_per_clip
            print(f"  Creating {num_clips} clips of {bars_per_clip} bars each")
            
            # Create clips
            for i in range(num_clips):
                start_bar = i * bars_per_clip
                end_bar = start_bar + bars_per_clip
                
                # Extract clip
                clip = multitrack[start_bar:end_bar]
                
                # Save clip
                clip_filename = f"{song_name}_clip_{start_bar}_{end_bar}.mid"
                clip_path = os.path.join(song_output_dir, clip_filename)
                clip.to_midi(clip_path)
                print(f"    Saved: {clip_filename}")
            
            # Handle remaining bars if any
            remaining_bars = total_bars % bars_per_clip
            if remaining_bars > 0:
                start_bar = num_clips * bars_per_clip
                end_bar = total_bars
                clip = multitrack[start_bar:end_bar]
                clip_filename = f"{song_name}_clip_{start_bar}_{end_bar}.mid"
                clip_path = os.path.join(song_output_dir, clip_filename)
                clip.to_midi(clip_path)
                print(f"    Saved remaining {remaining_bars} bars: {clip_filename}")
                
        except Exception as e:
            print(f"Error processing {midi_file}: {str(e)}")
            continue

if __name__ == "__main__":
    # Configuration
    input_folder = "../../song_midis/"  # Change this to your input folder
    output_folder = "../../song_clips/"  # Change this to your desired output folder
    bars_per_clip = 8  # Number of bars per clip
    
    # Process the folder
    process_midi_folder(input_folder, output_folder, bars_per_clip)