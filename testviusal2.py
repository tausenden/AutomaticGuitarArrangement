def visualize_guitar_tab(sequence):
    """
    Visualizes guitar tablature and finger positions from a sequence of (fret, finger) tuples.
    Each tuple contains two lists: fret positions and finger assignments for each string.
    
    Args:
        sequence: List of tuples, each containing two lists (fret positions and finger assignments)
    """
    # Initialize strings for both finger positions and fret positions
    strings = ['E|', 'B|', 'G|', 'D|', 'A|', 'E|']
    finger_strings = ['E|', 'B|', 'G|', 'D|', 'A|', 'E|']
    
    # Process each position in the sequence
    for i, (frets, fingers) in enumerate(sequence):
        # Add bar line every 4 positions
        if i > 0 and i % 4 == 0:
            for j in range(6):
                strings[j] += '|'
                finger_strings[j] += '|'
        
        # Process each string
        for string_idx in range(6):
            # Handle fret positions
            fret = frets[string_idx]
            if fret == -1:
                strings[string_idx] += '--'
            else:
                strings[string_idx] += str(fret).rjust(2)
            
            # Handle finger positions
            finger = fingers[string_idx]
            if finger == -1:
                finger_strings[string_idx] += '--'
            else:
                finger_strings[string_idx] += str(finger).rjust(2)
                
        # Add separator
        for j in range(6):
            strings[j] += '-'
            finger_strings[j] += '-'
    
    # Print results
    print("Finger positions:")
    for line in finger_strings:
        print(line)
    print("\nFret positions:")
    for line in strings:
        print(line)

# Example usage
tab_sequence = [([0, 6, -1, 0, -1, 0], [-1, 3, -1, -1, -1, -1]), 
                ([-1, 3, -1, -1, 7, 0], [-1, 3, -1, -1, 1, -1]),
                ([0, -1, -1, -1, -1, 0], [-1, -1, -1, -1, -1, -1]),
                ([-1, 5, 2, -1, 0, -1], [-1, 2, 4, -1, -1, -1]),
                ([-1, -1, 0, -1, -1, 0], [-1, -1, -1, -1, -1, -1]),
                ([0, 0, 0, 8, 0, -1], [-1, -1, -1, 3, -1, -1]),
                ([0, -1, 4, 0, -1, -1], [-1, -1, 4, -1, -1, -1]),
                ([-1, 0, 0, 4, 0, 0], [-1, -1, -1, 4, -1, -1])]

visualize_guitar_tab(tab_sequence)