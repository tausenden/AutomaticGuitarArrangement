from core import evaluate_arrangement
import os
import json
import numpy as np

def collect_and_summarize_metrics(arranged_root):
    """
    Collect all statistics.json files from song folders and create a summary JSON
    with all song metrics and average metrics across all songs.
    """
    all_metrics = []
    
    song_dirs = [d for d in os.listdir(arranged_root) if os.path.isdir(os.path.join(arranged_root, d))]
    
    for song_name in sorted(song_dirs):
        song_dir = os.path.join(arranged_root, song_name)
        stats_file = os.path.join(song_dir, f"{song_name}_statistics.json")
        
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r') as f:
                    metrics = json.load(f)
                    metrics['song_name'] = song_name
                    all_metrics.append(metrics)
                    print(f"Loaded metrics for {song_name}")
            except Exception as e:
                print(f"Error loading {stats_file}: {e}")
        else:
            print(f"Statistics file not found for {song_name}: {stats_file}")
    
    if not all_metrics:
        print("No metrics found to summarize")
        return
    
    # Calculate averages across all songs
    metric_keys = ['note_precision', 'rhythm_accuracy', 'chord_accuracy', 
                   'chord_name_accuracy', 'melody_accuracy', 'melody_correlation']
    
    averages = {}
    for key in metric_keys:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            averages[f'avg_{key}'] = round(np.mean(values), 4)
    # Create summary data
    summary = {
        'songs': all_metrics,
        'averages': averages,
        'total_songs': len(all_metrics)
    }
    
    # Save summary to arranged_root folder
    summary_file = os.path.join(arranged_root, "summary_metrics.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary metrics saved to {summary_file}")
    print(f"Average metrics across {len(all_metrics)} songs:")
    for key, value in averages.items():
        if key.startswith('avg_'):
            print(f"  {key[4:]}: {value}")
    
    return summary


def main():
    # Settings
    arranged_root = './arranged_songs_hmm'
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
            raise e
            print(f"Error evaluating {song_name}: {e}")
    
    # After evaluating all songs, collect and summarize metrics
    print("\n" + "="*50)
    print("Collecting and summarizing all metrics...")
    collect_and_summarize_metrics(arranged_root)


if __name__ == "__main__":
    main()