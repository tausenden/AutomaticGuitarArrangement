from core import GAreproducing, Guitar
from core.utils import GATab, GATabSeq
from core.utils.ga_export import export_ga_results
import json
import os

class GAreproPCExplainer(GAreproducing):
    """
    Subclass that exposes a detailed breakdown of the playability (PC) terms.
    It mirrors GAreproducing.calculate_playability but returns component terms as well.
    """
    def calculate_playability_terms(self, tablature):
        # Delegate to the base implementation to keep logic identical
        return super().calculate_playability_terms(tablature)

 

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
    default_midi = 'caihong_clip/caihong_clip_4_12.mid'
    ga_r = GAreproPCExplainer(midi_file_path=default_midi)

    comp_sum_played = 0.0
    comp_sum_fret = 0.0
    comp_sum_span = 0.0
    totals = []
    for bar_idx, bar in enumerate(bars):
        terms = ga_r.calculate_playability_terms(bar)
        totals.append(terms['total'])
        comp_sum_played += terms['played_strings_penalty']
        comp_sum_fret += terms['fret_distance_penalty']
        comp_sum_span += terms['span_difficulty']
        print(f"[TAB] Bar {bar_idx}: total={terms['total']:.4f}, played_strings_penalty={terms['played_strings_penalty']:.4f}, "
              f"fret_distance_penalty={terms['fret_distance_penalty']:.4f}, span_difficulty={terms['span_difficulty']:.4f}")

    n_bars = len(bars) if bars else 1
    avg_pc = sum(totals) / n_bars if totals else 0.0
    results = {
        'average_pc': round(avg_pc, 4),
        'average_terms': {
            'played_strings_penalty': round(comp_sum_played / n_bars, 4),
            'fret_distance_penalty': round(comp_sum_fret / n_bars, 4),
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
    not_important='caihong_clip/caihong_clip_4_12.mid'
    # Use the PC-explainer subclass; all other GA parts are unchanged
    ga_r = GAreproPCExplainer(
        midi_file_path=not_important
    )
    song_idx=0
    # Process each tab
    for tab_name, tab_info in tabs_data.items():
        bars = tab_info['bars']
        
        # Calculate PC, print component terms for each bar
        pc_values = []
        comp_sum_played = 0.0
        comp_sum_fret = 0.0
        comp_sum_span = 0.0
        for bar_idx, bar in enumerate(bars):
            terms = ga_r.calculate_playability_terms(bar)
            pc_values.append(terms['total'])
            comp_sum_played += terms['played_strings_penalty']
            comp_sum_fret += terms['fret_distance_penalty']
            comp_sum_span += terms['span_difficulty']
            print(f"Bar {bar_idx}: total={terms['total']:.4f}, played_strings_penalty={terms['played_strings_penalty']:.4f}, "
                  f"fret_distance_penalty={terms['fret_distance_penalty']:.4f}, span_difficulty={terms['span_difficulty']:.4f}")
        
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
                'span_difficulty': round(comp_sum_span / n_bars, 4)
            }
        }
        with open(os.path.join(output_dir, f"{tab_name}_pc.json"), 'w') as f:
            json.dump(pc_results, f, indent=2)

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

if __name__ == "__main__":
    cal_from_human()
    base_dir='./arranged_songs_hmm'
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