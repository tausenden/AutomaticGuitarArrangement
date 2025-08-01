import os
from sonata_utils import read_json, save_json, jpath, save_jsonl
from fretboard_util import Fretboard
import itertools

def main():
    generate_all_possible_positions()


def generate_all_possible_positions():
    """
    Generate all possible diagram on the guitar fretboard
    """
    positions = []
    HIGHEST_POSITION = 12

    ''' Open strings '''
    for string in range(1, 7):
        tab = ['x'] * 6
        tab[string - 1] = 0
        finger_used = [0] * 6

        config = infer_info(tab, finger_used)
        positions.append(config)

    ''' Single note positions '''
    for position in range(1, HIGHEST_POSITION + 1):
        for string in range(1, 7):
            for finger in range(1, 5):
                tab = ['x'] * 6
                tab[string - 1] = position - 1 + finger

                fingering = [0] * 6
                fingering[string - 1] = finger

                config = infer_info(tab, fingering)
                positions.append(config)

    ''' Open chord positions '''
    # Major
    chord_and_fingers = [
        (['x',3,2,0,1,0], [0,3,2,0,1,0]),   # C
        (['x','x',0,2,3,2], [0,0,0,2,3,1]),   # D
        ([0,2,2,1,0,0], [0,2,3,1,0,0]),   # E
        ([1,3,3,2,1,1], [1,3,4,2,1,1]), # F
        ([3,2,0,0,0,3], [3,2,0,0,0,4]),     # G
        (['x',0,2,2,2,0], [0,0,1,2,3,0]),       # A
        (['x','x',4,4,4,2], [0,0,2,3,4,1]),        # B
        (['x','x',5,5,4,3], [0,0,3,4,2,1]),   # Cm
        (['x','x',0,2,3,1], [0,0,0,2,3,1]),   # Dm
        ([0,2,2,0,0,0], [0,2,3,0,0,0]),   # Em
        ([1,3,3,1,1,1], [1,3,4,1,1,1]), # Fm
        (['x','x',5,3,3,3], [0,0,3,1,1,1]),     # Gm
        (['x',0,2,2,1,0], [0,0,2,3,1,0]),       # Am
        (['x',2,4,4,3,2], [0,1,3,4,2,1])        # Bm
    ]
    for chord, fingers in chord_and_fingers:
        # Find any possible subset of chord and fingering
        # Get all indices of non-muted strings
        string_indices = [i for i, fret in enumerate(chord) if fret != 'x']
        # For subset sizes 1 to n_strings-1 (exclude full chord, which is already added)
        for k in range(1, 6):
            for subset in itertools.combinations(string_indices, k):
                # Build new tab and fingering for this subset
                tab_sub = ['x'] * 6
                fingering_sub = [0] * 6
                for idx in subset:
                    tab_sub[idx] = chord[idx]
                    fingering_sub[idx] = fingers[idx]
                config = infer_info(tab_sub, fingering_sub)
                positions.append(config)
        # Add the full chord itself
        config = infer_info(chord, [f if isinstance(f, int) else 0 for f in fingers] + [0]*(6-len(fingers)))
        positions.append(config)

    ''' CAGED chord positions '''
    chord_and_fingers = [
        (['x',4,3,1,2,1], [0,4,3,1,2,1]),       # C shape
        (['x',1,3,3,3,'x'], [0,1,2,3,4,0]),     # A shape
        ([4,3,1,1,1,4], [3,2,1,1,1,4]),         # G shape
        ([1,3,3,2,1,1], [1,3,4,2,1,1]),         # E shape
        (['x','x',1,3,4,3], [0,0,1,2,4,3]),     # D shape
        (['x',4,2,1,2,'x'], [0,4,2,1,3,0]),       # C shape (minor)
        (['x',1,3,3,2,1], [0,1,3,4,2,1]),     # A shape (minor)
        (['x','x',1,4,4,4], [0,0,1,3,3,3]),         # G shape (minor)
        ([1,3,3,1,1,1], [1,3,4,1,1,1]),         # E shape (minor)
        (['x','x',1,3,4,2], [0,0,1,3,4,2]),         # D shape (minor)
    ]
    for chord, fingers in chord_and_fingers:
        for position in range(1, HIGHEST_POSITION + 1):
            # Adjust the tab to the current position
            tab = [i + position - 1 if isinstance(i, int) else 'x' for i in chord]
            fingering = fingers 

            # Get all possible subsets of the chord
            string_indices = [i for i, fret in enumerate(tab) if fret != 'x']
            for k in range(1, 6):
                for subset in itertools.combinations(string_indices, k):
                    # Build new tab and fingering for this subset
                    tab_sub = ['x'] * 6
                    fingering_sub = [0] * 6
                    for idx in subset:
                        tab_sub[idx] = tab[idx]
                        fingering_sub[idx] = fingering[idx]
                    config = infer_info(tab_sub, fingering_sub)
                    positions.append(config)
            # Add the full chord itself
            config = infer_info(tab, [f if isinstance(f, int) else 0 for f in fingering] + [0]*(6-len(fingering)))
            positions.append(config)

    # Deduplicate positions based on tab and fingering
    unique_positions = {}
    for p in positions:
        # Convert tab and fingering to tuple for hashing (handle 'x' as string)
        tab_tuple = tuple(p['tab'])
        fingering_tuple = tuple(p['fingering'])
        key = (tab_tuple, fingering_tuple)
        if key not in unique_positions:
            unique_positions[key] = p
    positions = list(unique_positions.values())

    print(len(positions))
    save_fp = jpath('diagram_config', 'possible_positions.jsonl')
    save_jsonl(positions, save_fp)
    
    
    return positions

fretboard = Fretboard()
def infer_info(tab, finger_used):
    '''
    Infer information from the generated positions
        pitches: a list of integers
        index_pos: lowest non-zero position on the tab. For open string, it is 0 (means can be in any position)
        string: a list of used string IDs (1~6)	(can be inferred)
        fingers: number of fingers used (1~4)	(can be inferred)
    return the full config
    '''
    min_fret = 21
    for i in tab:
        if i != 'x' and i != 0:
            min_fret = min(min_fret, i)
    if min_fret == 21:
        min_fret = 0
    index_pos = min_fret

    pitches = []
    for i, fret in enumerate(tab):
        if fret != 'x':
            string_id = 6 - i - 1
            pitch = int(fretboard.get_pitch(string_id, fret))
            pitches.append(pitch)
    pitches.sort()
    
    string = [6-i for i, fret in enumerate(tab) if fret != 'x']

    # count non-zero fingers
    non_zero_fingers = [f for f in finger_used if f != 0]
    fingers = len(non_zero_fingers)

    ret = {
        'tab': tab,
        'fingering': finger_used,
        'index_pos': index_pos,
        'pitches': pitches,
        'string_ids': string,
        'n_fingers': fingers,
    }
    return ret

if __name__ == "__main__":
    main()