from core import evaluate_arrangement
import os
import json

def main():
    # Settings
    arranged_root = './arranged_songs_GA_r'
    original_roots = ['./song_midis']
    resolution = 16

    if not os.path.isdir(arranged_root):
        print(f"Arranged root not found: {arranged_root}")
        return

    song_dirs = [d for d in os.listdir(arranged_root) if os.path.isdir(os.path.join(arranged_root, d))]
    if not song_dirs:
        print(f"No arranged songs found under {arranged_root}")
        return

    for song_name in sorted(song_dirs):
        song_dir = os.path.join(arranged_root, song_name)

        # Find arranged MIDI in the song's folder
        arranged_candidates = [f for f in os.listdir(song_dir) if f.lower().endswith(('.mid', '.midi'))]
        if not arranged_candidates:
            print(f"Skipping {song_name}: no arranged MIDI found in {song_dir}")
            continue

        preferred = next((f for f in arranged_candidates if os.path.splitext(f)[0] == song_name), None)
        arranged_midi_path = os.path.join(song_dir, preferred or arranged_candidates[0])

        # Locate original MIDI in provided roots
        original_midi_path = None
        for root in original_roots:
            for ext in ('.mid', '.midi'):
                candidate = os.path.join(root, f"{song_name}{ext}")
                if os.path.isfile(candidate):
                    original_midi_path = candidate
                    break
            if original_midi_path:
                break

        if not original_midi_path:
            print(f"Skipping {song_name}: original MIDI not found in {original_roots}")
            continue

        # Evaluate with simple error handling
        try:
            print(f"Evaluating {song_name}:\n  original: {original_midi_path}\n  arranged: {arranged_midi_path}")
            evaluate_arrangement(
                original_midi_path=original_midi_path,
                arranged_midi_path=arranged_midi_path,
                output_dir=song_dir,
                song_name=song_name,
                resolution=resolution,
                export_separate=False
            )
            print(f"Done evaluating {song_name}")
        except Exception as e:
            print(f"Error evaluating {song_name}: {e}")

if __name__ == "__main__":
    main()