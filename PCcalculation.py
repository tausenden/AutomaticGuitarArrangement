from core import GAreproducing, Guitar
from core.utils import GATab, GATabSeq
from core.utils.ga_export import export_ga_results
import json
import os
import numpy as np

class GAreproPCExplainer(GAreproducing):
    """
    Subclass that exposes a detailed breakdown of the playability (PC) terms.
    It mirrors GAreproducing.calculate_playability but returns component terms as well.
    """
    def calculate_playability_terms(self, tablature):
        # Delegate to the base implementation to keep logic identical

        terms = super().calculate_playability_terms(tablature)
        index_penalty = 0
        active_positions = [pos for pos, chord in enumerate(tablature) if any(fret != -1 for fret in chord)]
        # Compute minimal index-finger movement path across positions with pressed frets (>0)
        press_positions = [pos for pos in active_positions if any(f > 0 for f in tablature[pos])]
        if len(press_positions) >= 2:
            # Build candidate index positions (unique positive fret values) for each pressed position
            index_candidates_per_pos = []
            for pos in press_positions:
                pressed_frets = [f for f in tablature[pos] if f > 0]
                if pressed_frets:
                    min_pressed = min(pressed_frets)
                    max_pressed = max(pressed_frets)
                    # Allowed index positions i must satisfy: for each pressed fret f, 0 <= f - i <= 3
                    # => i in [f-3, f] for each f; take intersection across all f
                    i_low = max(0, max(f - 3 for f in pressed_frets))
                    i_high = min_pressed
                    if i_low <= i_high:
                        candidates = list(range(i_low, i_high + 1))
                    else:
                        # No single hand position covers all frets within 4-fret span.
                        # Provide two plausible anchors to keep DP feasible and implicitly penalize span:
                        # - Anchor at lowest fret (barre at min)
                        # - Anchor so highest fret is pinky (max-3)
                        candidates = sorted({min_pressed, max(0, max_pressed - 3)})
                else:
                    candidates = [0]
                index_candidates_per_pos.append(candidates)

            # Dynamic programming to minimize total movement of index finger
            prev_pos = press_positions[0]
            prev_dp = {f: 0.0 for f in index_candidates_per_pos[0]}
            for idx in range(1, len(press_positions)):
                curr_pos = press_positions[idx]
                interval = curr_pos - prev_pos
                curr_candidates = index_candidates_per_pos[idx]
                curr_dp = {}
                for curr_fret in curr_candidates:
                    # Transition from any previous index fret
                    best_cost = float('inf')
                    for prev_fret, prev_cost in prev_dp.items():
                        move_cost = abs(curr_fret - prev_fret)
                        if interval > 6:
                            move_cost = move_cost / ((interval - 6) ** 0.5)
                        best_cost = min(best_cost, prev_cost + move_cost)
                    curr_dp[curr_fret] = best_cost
                prev_dp = curr_dp
                prev_pos = curr_pos

            min_total_movement = min(prev_dp.values()) if prev_dp else 0.0
            index_penalty -= min_total_movement

        terms['index_penalty'] = index_penalty

        return terms


def collect_and_summarize_pc_metrics(pc_root):
    """
    Collect all pc_from_tab.json files from song folders and create a summary JSON
    with all song PC metrics and average metrics across all songs.
    """
    all_metrics = []
    
    song_dirs = [d for d in os.listdir(pc_root) if os.path.isdir(os.path.join(pc_root, d))]
    
    for song_name in sorted(song_dirs):
        song_dir = os.path.join(pc_root, song_name)
        pc_file = os.path.join(song_dir, f"{song_name}_pc_from_tab.json")
        
        if os.path.exists(pc_file):
            try:
                with open(pc_file, 'r') as f:
                    metrics = json.load(f)
                    metrics['song_name'] = song_name
                    all_metrics.append(metrics)
                    print(f"Loaded PC metrics for {song_name}")
            except Exception as e:
                print(f"Error loading {pc_file}: {e}")
        else:
            print(f"PC file not found for {song_name}: {pc_file}")
    
    if not all_metrics:
        print("No PC metrics found to summarize")
        return
    
    # Calculate averages across all songs
    metric_keys = ['average_pc']
    term_keys = ['played_strings_penalty', 'fret_distance_penalty', 'index_penalty', 'span_difficulty']
    
    averages = {}
    for key in metric_keys:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            averages[f'avg_{key}'] = round(np.mean(values), 4)
    
    # Calculate averages for term metrics
    for key in term_keys:
        values = [m['average_terms'][key] for m in all_metrics if 'average_terms' in m and key in m['average_terms']]
        if values:
            averages[f'avg_{key}'] = round(np.mean(values), 4)
    
    # Create summary data
    summary = {
        'songs': all_metrics,
        'averages': averages,
        'total_songs': len(all_metrics)
    }
    
    # Save summary to pc_root folder
    summary_file = os.path.join(pc_root, "summary_pc_metrics.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary PC metrics saved to {summary_file}")
    print(f"Average PC metrics across {len(all_metrics)} songs:")
    for key, value in averages.items():
        print(f"  {key}: {value}")
    
    return summary


def _split_row_segments(line):
    """Split a combined row into per-bar segments using the 3-space separator used by TabSeq.__str__."""
    # Ensure consistent splitting even if trailing spaces
    parts = line.rstrip('\n').split('   ')
    # Normalize right-strip each segment to avoid dangling spaces
    return [p.rstrip() for p in parts]

def _parse_bar_segment_string_to_tokens(segment):
    """Parse a single string's segment (e.g., "-- -- 12 -- ...") into a list of fret ints (-1 for rests)."""
    tokens = []
    for tok in segment.strip().split():
        if tok == '--':
            tokens.append(-1)
        else:
            # Right-justified numbers like " 7" -> int
            tokens.append(int(tok))
    return tokens

def bars_from_ga_tab(tab_path):
    """
    Parse a GA-exported .tab file (from GATabSeq.__str__) back into list-of-lists bar data.
    Returns: List[Bar], each Bar is List[position][6] of fret ints.
    """
    with open(tab_path, 'r') as f:
        lines = [ln.rstrip('\n') for ln in f]

    bars_tabs = []
    block = []
    for ln in lines + ['']:
        if ln.strip() == '':
            if block:
                # Expect: first line chord names (optional), last 6 lines are strings
                string_lines = block[-6:] if len(block) >= 6 else []
                if len(string_lines) == 6:
                    # Split each string line into per-bar segments
                    split_lines = [ _split_row_segments(sl) for sl in string_lines ]
                    num_segments = max(len(seglist) for seglist in split_lines)
                    for seg_idx in range(num_segments):
                        # Collect the 6 string segments for this bar; pad with '' if missing
                        string_segs = [ (split_lines[s][seg_idx] if seg_idx < len(split_lines[s]) else '') for s in range(6) ]
                        # Parse tokens per string
                        per_string_tokens = [ _parse_bar_segment_string_to_tokens(seg) for seg in string_segs ]
                        if not per_string_tokens or any(len(toks) == 0 for toks in per_string_tokens):
                            continue
                        # Ensure all strings have the same number of positions
                        n_pos = max(len(toks) for toks in per_string_tokens)
                        # Pad shorter strings with rests to match positions
                        per_string_tokens = [ toks + [-1] * (n_pos - len(toks)) for toks in per_string_tokens ]
                        # Transpose into positions: each position is 6-element list [s0..s5]
                        bar_positions = []
                        for pos in range(n_pos):
                            chord = [ per_string_tokens[s][pos] for s in range(6) ]
                            bar_positions.append(chord)
                        bars_tabs.append(bar_positions)
                block = []
        else:
            block.append(ln)
    return bars_tabs

def cal_from_ga(tab_path, output_dir=None):
    """
    Read a GA-exported .tab file, compute average PC and average term values, and save JSON next to the tab.
    Returns (results_dict, json_path).
    """
    bars = bars_from_ga_tab(tab_path)
    # Use any valid MIDI to initialize GA context; fallback to a placeholder by reusing an existing small clip
    # Since we only need PC (tab-only), instantiate with a known clip; replace with your project default if needed
    default_midi = 'song_midis/caihong.mid'
    ga_r = GAreproPCExplainer(midi_file_path=default_midi)

    comp_sum_played = 0.0
    comp_sum_fret = 0.0
    comp_sum_span = 0.0
    comp_sum_index= 0.0
    totals = []
    for bar_idx, bar in enumerate(bars):
        terms = ga_r.calculate_playability_terms(bar)
        total = terms['played_strings_penalty'] + terms['span_difficulty'] + terms['index_penalty']
        totals.append(total)
        comp_sum_played += terms['played_strings_penalty']
        comp_sum_fret += terms['fret_distance_penalty']
        comp_sum_span += terms['span_difficulty']
        comp_sum_index += terms['index_penalty']
        print(f"[TAB] Bar {bar_idx}: total={total:.4f}, played_strings_penalty={terms['played_strings_penalty']:.4f}, "
              f"fret_distance_penalty={terms['fret_distance_penalty']:.4f}, span_difficulty={terms['span_difficulty']:.4f}, index_penalty={terms['index_penalty']:.4f}")

    n_bars = len(bars) if bars else 1
    avg_pc = sum(totals) / n_bars if totals else 0.0
    results = {
        'average_pc': round(avg_pc, 4),
        'average_terms': {
            'played_strings_penalty': round(comp_sum_played / n_bars, 4),
            'fret_distance_penalty': round(comp_sum_fret / n_bars, 4),
            'index_penalty': round(comp_sum_index / n_bars, 4),
            'span_difficulty': round(comp_sum_span / n_bars, 4)      
        }
    }

    if output_dir is None:
        output_dir = os.path.dirname(tab_path)
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(tab_path))[0]
    json_path = os.path.join(output_dir, f"{base}_pc_from_tab.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    return results, json_path

def convert_to_ga_tab_seq(bars_data):
    """
    Convert list of bars (list of list format) to GATabSeq
    Args:
        bars_data: List of bars, each bar is a list of chord positions (list of lists)
    Returns:
        GATabSeq: Converted GATabSeq object
    """
    ga_tabs = []
    for bar_data in bars_data:
        # Create GATab from bar_data (list of lists format)
        ga_tab = GATab(bar_data=bar_data)
        ga_tabs.append(ga_tab)
    
    return GATabSeq(tab_list=ga_tabs)

def cal_from_human():
    tempos=[80,80,110,75,90]
    # Read guitar tab from JSON file
    with open('human_guitar_tab.json', 'r') as file:
        tabs_data = json.load(file)
    not_important='song_midis/caihong.mid'
    # Use the PC-explainer subclass; all other GA parts are unchanged
    ga_r = GAreproPCExplainer(
        midi_file_path=not_important
    )
    song_idx=0
    all_metrics = []
    # Process each tab
    for tab_name, tab_info in tabs_data.items():
        bars = tab_info['bars']
        
        # Calculate PC, print component terms for each bar
        pc_values = []
        comp_sum_played = 0.0
        comp_sum_fret = 0.0
        comp_sum_span = 0.0
        comp_sum_index = 0.0
        for bar_idx, bar in enumerate(bars):
            terms = ga_r.calculate_playability_terms(bar)
            pc_values.append(terms['total'])
            comp_sum_played += terms['played_strings_penalty']
            comp_sum_fret += terms['fret_distance_penalty']
            comp_sum_span += terms['span_difficulty']
            comp_sum_index += terms['index_penalty']
            print(f"Bar {bar_idx}: total={terms['total']:.4f}, played_strings_penalty={terms['played_strings_penalty']:.4f}, "
                  f"fret_distance_penalty={terms['fret_distance_penalty']:.4f}, span_difficulty={terms['span_difficulty']:.4f}, index_penalty={terms['index_penalty']:.4f}")
        
        avg_pc = sum(pc_values) / len(pc_values) if pc_values else 0.0
        print(f"{tab_name}: average PC = {avg_pc:.4f}")
        
        # Convert to GATabSeq and export
        ga_tab_seq = convert_to_ga_tab_seq(bars)
        
        # Create output directory
        output_dir = f"human_guitar/{tab_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save only the final average PC and average component terms to JSON
        n_bars = len(bars) if bars else 1
        pc_results = {
            'average_pc': round(avg_pc, 4),
            'average_terms': {
                'played_strings_penalty': round(comp_sum_played / n_bars, 4),
                'fret_distance_penalty': round(comp_sum_fret / n_bars, 4),
                'span_difficulty': round(comp_sum_span / n_bars, 4),
                'index_penalty': round(comp_sum_index / n_bars, 4)
            }
        }
        with open(os.path.join(output_dir, f"{tab_name}_pc.json"), 'w') as f:
            json.dump(pc_results, f, indent=2)

        # Collect for summary
        metrics_entry = dict(pc_results)
        metrics_entry['song_name'] = tab_name
        all_metrics.append(metrics_entry)

        # Export tab, midi, wav
        file_paths = export_ga_results(
            ga_tab_seq=ga_tab_seq,
            resolution=16,
            song_name=tab_name,
            tempo=tempos[song_idx],
            output_dir=output_dir,
            sf2_path='resources/Tyros Nylon.sf2',
            verbose=False
        )
        song_idx+=1
        print()

    # Write overall summary for human tabs
    if all_metrics:
        metric_keys = ['average_pc']
        term_keys = ['played_strings_penalty', 'fret_distance_penalty',  'index_penalty', 'span_difficulty']

        averages = {}
        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                averages[f'avg_{key}'] = round(np.mean(values), 4)

        for key in term_keys:
            values = [m['average_terms'][key] for m in all_metrics if 'average_terms' in m and key in m['average_terms']]
            if values:
                averages[f'avg_{key}'] = round(np.mean(values), 4)

        summary = {
            'songs': all_metrics,
            'averages': averages,
            'total_songs': len(all_metrics)
        }

        human_root = 'human_guitar'
        os.makedirs(human_root, exist_ok=True)
        summary_file = os.path.join(human_root, 'summary_pc_metrics.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary PC metrics for human tabs saved to {summary_file}")

if __name__ == "__main__":
    # cal_from_human()
    base_dir='./arranged_songs_GA_i'
    if os.path.isdir(base_dir):
        for song_name in sorted(os.listdir(base_dir)):
            song_dir = os.path.join(base_dir, song_name)
            if not os.path.isdir(song_dir):
                continue
            tab_files = [fn for fn in os.listdir(song_dir) if fn.endswith('.tab')]
            if not tab_files:
                continue
            tab_path = os.path.join(song_dir, tab_files[0])
            cal_from_ga(tab_path)
        
        # After processing all songs, collect and summarize PC metrics
        print("\n" + "="*50)
        print("Collecting and summarizing all PC metrics...")
        collect_and_summarize_pc_metrics(base_dir)