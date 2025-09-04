from core import GAreproducing, GAimproved, Guitar
from core.utils import GATab, GATabSeq
from core.utils.ga_export import export_ga_results
from remi_z import MultiTrack
import json
import os

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

def main():
    tempos=[80,80,110,75]
    # Read guitar tab from JSON file
    with open('human_guitar_tab.json', 'r') as file:
        tabs_data = json.load(file)
    
    ga_r = GAreproducing(
        midi_file_path='caihong_clip/caihong_clip_4_12.mid'
    )
    ga_i = GAimproved(
        midi_file_path='caihong_clip/caihong_clip_4_12.mid'
    )
    song_idx=0
    # Process each tab
    for tab_name, tab_info in tabs_data.items():
        bars = tab_info['bars']
        
        # Calculate average PC
        pc_values_r = []
        pc_values_i = []
        nwc_values_r = []
        nwc_values_i = []
        ncc_values_r = []
        ncc_values_i = []
        rp_values_i = []
        for bar_idx, bar in enumerate(bars):
            original_midi_pitches = ga_r.bars_data[bar_idx]['original_midi_pitches']
            chords = ga_r.bars_data[bar_idx]['chords']
            pc_r = ga_r.calculate_playability(bar)
            pc_values_r.append(pc_r)
            hand=ga_i.get_finger_assignment(bar)
            candi={'hand_candi':hand,'tab_candi':bar}
            pc_i = ga_i.calculate_playability(candi)
            pc_values_i.append(pc_i)
            nwc_r = ga_r.calculate_NWC(bar,original_midi_pitches)
            nwc_i = ga_i.calculate_NWC(candi,original_midi_pitches)
            nwc_values_r.append(nwc_r)
            nwc_values_i.append(nwc_i)
            ncc_r = ga_r.calculate_NCC(bar,original_midi_pitches)
            ncc_i = ga_i.calculate_NCC(candi,original_midi_pitches,chords)
            ncc_values_r.append(ncc_r)
            ncc_values_i.append(ncc_i)
            rp_i = ga_i.calculate_RP(candi,bar_idx)
            rp_values_i.append(rp_i)
        
        avg_pc_r = sum(pc_values_r) / len(pc_values_r)
        avg_pc_i = sum(pc_values_i) / len(pc_values_i)
        avg_nwc_r = sum(nwc_values_r) / len(nwc_values_r)
        avg_nwc_i = sum(nwc_values_i) / len(nwc_values_i)
        avg_ncc_r = sum(ncc_values_r) / len(ncc_values_r)
        avg_ncc_i = sum(ncc_values_i) / len(ncc_values_i)
        avg_rp_i = sum(rp_values_i) / len(rp_values_i)
        print(f"{tab_name}:\n")
        print(f"GA_repro:")
        print(f" PC:{avg_pc_r:.4f}\n NWC:{avg_nwc_r:.4f}\n NCC:{avg_ncc_r:.4f}")
        print(f"\nGA_improved:")
        print(f" PC:{avg_pc_i:.4f}\n NWC:{avg_nwc_i:.4f}\n NCC:{avg_ncc_i:.4f}\n RP:{avg_rp_i:.4f}")
        
        # Convert to GATabSeq and export
        ga_tab_seq = convert_to_ga_tab_seq(bars)
        
        # Create output directory
        output_dir = f"human_guitar/{tab_name}"
        os.makedirs(output_dir, exist_ok=True)
        
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
    main()