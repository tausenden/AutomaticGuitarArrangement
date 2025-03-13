def visualize_guitar_tab(sequence):
    """
    Visualizes guitar tablature and finger positions from a sequence of (fret, finger) tuples or just fret positions.
    Each tuple may contain either two lists (fret positions and finger assignments) or just fret positions.
    
    Args:
        sequence: List of tuples, each containing either one or two lists (fret positions and optionally finger assignments)
    """
    # Initialize strings for fret positions
    strings = ['E|', 'B|', 'G|', 'D|', 'A|', 'E|']
    finger_strings = ['E|', 'B|', 'G|', 'D|', 'A|', 'E|']
    
    # Process each position in the sequence
    for i, entry in enumerate(sequence):
        # Check if the entry is a tuple with finger positions
        if isinstance(entry, tuple):
            frets, fingers = entry
        else:
            frets = entry
            fingers = None
        
        # Add bar line every 8 positions
        if i > 0 and i % 8 == 0:
            for j in range(6):
                strings[j] += '|'
                if fingers is not None:
                    finger_strings[j] += '|'
        
        # Process each string
        for string_idx in range(6):
            # Handle fret positions
            fret = frets[string_idx]
            if fret == -1:
                strings[string_idx] += '--'
            else:
                strings[string_idx] += str(fret).rjust(2)
            
            # Handle finger positions if available
            if fingers is not None:
                finger = fingers[string_idx]
                if finger == -1:
                    finger_strings[string_idx] += '--'
                else:
                    finger_strings[string_idx] += str(finger).rjust(2)
                    
        # Add separator
        for j in range(6):
            strings[j] += '-'
            if fingers is not None:
                finger_strings[j] += '-'
    
    # Print results
    if fingers is not None:
        print("Finger positions:")
        for line in finger_strings:
            print(line)
        print()
    
    print("Fret positions:")
    for line in strings:
        print(line)


# Example usage
sequence_with_fingers = [([3, 2, 0, 0, 0, 3], [2, 1, -1, -1, -1, 3]), ([5, 3, 2, 0, 1, 0], [3, 2, 1, -1, 1, -1])]
sequence_without_fingers = [
    [3, 2, 0, 0, 0, 3],
    [5, 3, 2, 0, 1, 0]
]

print("With finger positions:")
visualize_guitar_tab(sequence_with_fingers)

print("\nWithout finger positions:")
visualize_guitar_tab(sequence_without_fingers)
