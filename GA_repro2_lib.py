import numpy as np
import random
from GAutils import Guitar, pitch2name, visualize_guitar_tab
from remi_z import MultiTrack

class GAreproducing:
    def __init__(self, guitar=None, mutation_rate=0.03, population_size=300, 
                 generations=100, num_strings=6, max_fret=20,
                 midi_file_path=None):
        """
        Implementation of the genetic algorithm for guitar arrangements with position-based approach.
        
        Parameters:
        -----------
        guitar : Guitar
            Guitar instance for fretboard calculations
        mutation_rate : float
            Probability of mutation (0.03 as in the paper)
        population_size : int
            Size of the population (300 as in the paper)
        generations : int
            Number of generations to run
        num_strings : int
            Number of strings on the guitar (typically 6)
        max_fret : int
            Maximum fret position to consider
        midi_file_path : str
            Path to MIDI file to process
        """
        if guitar is None:
            self.guitar = Guitar()
        else:
            self.guitar = guitar
        
        # Algorithm parameters
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.generations = generations
        self.num_strings = num_strings
        self.max_fret = max_fret
        
        # Crossover rate - 60% as in the paper
        self.crossover_rate = 0.6
        
        # Component weights for fitness function
        self.weight_PC = 1.0    # Playability Component
        self.weight_NWC = 1.0   # Note Weight Component
        self.weight_NCC = 1.0   # Notes in Chord Component
        self.note_categories = {}
        # Define note category weights
        self.category_weights = {
            'melody': 2.0,
            'harmony': 1.0,
        }
        
        # Ensure we have a MIDI file to process
        if not midi_file_path:
            raise ValueError("MIDI file path is required")
            
        # Load MIDI file using MultiTrack
        mt = MultiTrack.from_midi(midi_file_path)
        
        self.bars_data = []
        self.target_melody_list = []
        self.target_chord_list = []
        
        # Extract melody and chord data for each bar
        for bar in mt.bars:
            # Initialize an array of 48 positions, each with -1 (no note)
            positions = [-1] * 48
            
            # First extract the melody notes for categorization
            melody_notes = bar.get_melody('hi_note')
            melody_positions = set()
            for note in melody_notes:
                if 0 <= note.onset < 48:
                    melody_positions.add(note.onset)
            
            # Now get ALL notes from the bar
            all_notes = bar.get_all_notes(include_drum=False)
            
            # Fill positions array with ALL notes, preserving their position
            for note in all_notes:
                pos = note.onset
                if 0 <= pos < 48:
                    # If this position is in melody_positions, it's a melody note
                    # Otherwise it's a harmony note (we'll use the highest pitch at each position)
                    if pos in melody_positions:
                        # For melody positions, prioritize the melody note
                        if positions[pos] == -1 or note.pitch > positions[pos]:
                            positions[pos] = note.pitch
                    else:
                        # For harmony positions, always use the highest note
                        if positions[pos] == -1 or note.pitch > positions[pos]:
                            positions[pos] = note.pitch
            
            # Extract chord information
            chords = bar.get_chord()
            chord_name = chords[0][0] if chords and chords[0] else 'C'
            
            # Store the data
            self.bars_data.append({
                'positions': positions,
                'chord': chord_name
            })
            
            # Store melody and chord in lists as well
            # For backward compatibility if needed
            melody = []
            for note in melody_notes:
                melody.append(note.pitch)
            self.target_melody_list.append(melody)
            self.target_chord_list.append(chord_name)
            
            # Mark which positions are melody notes vs. harmony notes
            bar_categories = {}
            for note in all_notes:
                pos = note.onset
                if 0 <= pos < 48:
                    if pos in melody_positions:
                        bar_categories[pos] = 'melody'
                    else:
                        bar_categories[pos] = 'harmony'
            
            # Update the note categories dictionary
            self.note_categories.update(bar_categories)
    
    def initialize_population(self, bar_idx):
        """Initialize a random population of tablatures for a specific bar"""
        positions = self.bars_data[bar_idx]['positions']
        population = []
        
        for _ in range(self.population_size):
            tablature = []
            for pos in range(48):
                if positions[pos] == -1:
                    # No note at this position
                    tablature.append([-1, -1, -1, -1, -1, -1])
                else:
                    # Create a random chord position where at least one string is played
                    chord = []
                    for _ in range(self.num_strings):
                        if random.random() < 0.7:  # 70% chance for not played
                            chord.append(-1)
                        else:
                            # Random fret
                            fret = random.randint(0, self.max_fret)
                            chord.append(fret)
                    
                    # Ensure at least one string is played
                    if all(fret == -1 for fret in chord):
                        string_idx = random.randint(0, self.num_strings - 1)
                        chord[string_idx] = random.randint(0, self.max_fret)
                    
                    tablature.append(chord)
            
            population.append(tablature)
        
        return population
    
    def calculate_playability(self, tablature):
        """
        Calculate the Playability Component (PC)
        
        This assesses hand/finger movement difficulty and hand/finger manipulation.
        A higher value indicates an easier tablature to play.
        """
        # Filter out positions with no notes
        active_positions = [pos for pos, chord in enumerate(tablature) 
                           if any(fret != -1 for fret in chord)]
        
        if not active_positions:
            return 0  # No notes to evaluate
            
        # Extract only the active chords for evaluation
        active_chords = [tablature[pos] for pos in active_positions]
        
        # === Hand Movement Difficulty ===
        
        # Penalty for the total number of pressed strings
        pressed_strings_penalty = -sum(sum(1 for fret in chord if fret > 0) for chord in active_chords)
        
        # Penalty for fret-wise distance between adjacent active positions
        fret_distance_penalty = 0
        
        for i in range(1, len(active_positions)):
            prev_chord = tablature[active_positions[i-1]]
            curr_chord = tablature[active_positions[i]]
            
            prev_frets = [f for f in prev_chord if f > 0]
            curr_frets = [f for f in curr_chord if f > 0]
            
            if prev_frets and curr_frets:
                prev_avg = sum(prev_frets) / len(prev_frets)
                curr_avg = sum(curr_frets) / len(curr_frets)
                fret_distance_penalty -= abs(curr_avg - prev_avg)
        
        # Reward for open strings (easier to play)
        open_string_reward = sum(sum(1 for fret in chord if fret == 0) for chord in active_chords)
        
        # === Hand Manipulation Difficulty ===
        
        # Analyze chord shapes (easier vs harder to form)
        chord_difficulty = 0
        
        for chord in active_chords:
            pressed = [fret for fret in chord if fret > 0]
            
            if not pressed:
                continue
                
            # Determine span of the chord
            min_fret = min(pressed)
            max_fret = max(pressed)
            span = max_fret - min_fret + 1
            
            # Wider spans are harder to play
            if span > 4:
                chord_difficulty -= (span - 4) * 0.5
                
            # Detect if it's likely a barre chord
            same_fret_count = sum(1 for fret in chord if fret == min_fret and fret > 0)
            if same_fret_count >= 3 and span <= 5:
                chord_difficulty += 1.0  # Barre chords are efficient
        
        # Combine all factors
        playability = (pressed_strings_penalty + fret_distance_penalty + 
                      open_string_reward + chord_difficulty)
        
        # Normalize by number of active chords
        return playability
    
    def calculate_NWC(self, tablature, positions):
        """
        Calculate the Note Weight Component (NWC)
        
        For melody positions: measures pitch distance
        For harmony positions: checks for any note being played
        Uses weights based on note category (melody vs. harmony)
        """
        total_score = 0
        
        for pos, (chord, target) in enumerate(zip(tablature, positions)):
            # Skip if the original doesn't have a note at this position
            if target == -1:
                continue
                
            # Convert chord to MIDI notes
            chord_dict = {j+1: fret for j, fret in enumerate(chord)}
            midi_notes = self.guitar.get_chord_midi(chord_dict)
            midi_notes = [note for note in midi_notes if note != -1]
            
            # Check if any note is being played
            if not midi_notes:
                # Penalize positions where no notes are played
                total_score -= 10
                continue
            
            # Get category weight for this position
            category = self.note_categories.get(pos, 'harmony')
            weight = self.category_weights.get(category, 1.0)
            highest_note = max(midi_notes)
            pitch_distance = abs(highest_note - target)
            if pitch_distance == 0:
                # Perfect match
                total_score += weight * 2
            else:
                # Penalty proportional to distance
                total_score -= pitch_distance * (weight)
        
        return total_score
    
    def calculate_NCC(self, tablature):
        """
        Calculate the Notes in Chord Component (NCC) as in the paper.
        
        This measures the distribution of notes across the arrangement
        with diminishing returns for additional notes in each chord.
        Returns raw score without normalization.
        """
        total_chord_bonus = 0
        
        for chord in tablature:
            # Skip positions with no notes
            if all(fret == -1 for fret in chord):
                continue
            
            # Count notes in this chord/position
            chord_dict = {j+1: fret for j, fret in enumerate(chord)}
            midi_notes = self.guitar.get_chord_midi(chord_dict)
            notes_count = sum(1 for note in midi_notes if note != -1)
            
            # Calculate bonus with exponential function (diminishing returns)
            if notes_count > 0:
                chord_bonus = 1 - np.exp(-0.5 * notes_count)
                total_chord_bonus += chord_bonus
        
        return total_chord_bonus
    
    def fitness(self, tablature, positions):
        """Calculate the overall fitness of a tablature"""
        pc = self.calculate_playability(tablature)
        nwc = self.calculate_NWC(tablature, positions)
        ncc = self.calculate_NCC(tablature)
        
        # Apply weights to each component
        weighted_fitness = (
            pc * self.weight_PC + 
            nwc * self.weight_NWC + 
            ncc * self.weight_NCC
        )
        
        return weighted_fitness
    
    def two_point_crossover(self, parent1, parent2):
        """Perform two-point crossover between parents"""
        if len(parent1) < 3:
            return parent1.copy(), parent2.copy()
            
        point1, point2 = sorted(random.sample(range(len(parent1)), 2))
        
        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        
        return child1, child2
    
    def mutate(self, tablature, positions):
        """
        Simple random mutation.
        With probability mutation_rate, randomly change one string in a position.
        """
        for pos in range(len(tablature)):
            if random.random() < self.mutation_rate:
                # Only mutate positions that should have a note
                if positions[pos] != -1:
                    # Select a random string
                    string_idx = random.randint(0, self.num_strings - 1)
                    
                    # Apply random mutation
                    if random.random() < 0.3:  # 30% chance to not play this string
                        tablature[pos][string_idx] = -1
                    else:  # 70% chance to play a random fret
                        tablature[pos][string_idx] = random.randint(0, self.max_fret)
                    
                    # Ensure at least one string is played at positions that need a note
                    if all(fret == -1 for fret in tablature[pos]):
                        string_idx = random.randint(0, self.num_strings - 1)
                        tablature[pos][string_idx] = random.randint(0, self.max_fret)
        
        return tablature
    
    def run_single(self, bar_idx):
        """Run the genetic algorithm for a single bar with the best candidates as parents, without tournament selection"""
        bar_data = self.bars_data[bar_idx]
        positions = bar_data['positions']
        
        # Initialize population
        population = self.initialize_population(bar_idx)
        
        # Track best individual across all generations
        best_tablature = None
        best_fitness = float('-inf')
        
        # Main generational loop
        for generation in range(self.generations):
            # Calculate fitness for each individual
            fitnesses = [self.fitness(tab, positions) for tab in population]
            
            # Get indices sorted by fitness (highest first)
            sorted_indices = np.argsort(fitnesses)[::-1]
            
            # Find best in this generation
            gen_best_idx = sorted_indices[0]
            gen_best_fitness = fitnesses[gen_best_idx]
            gen_best_tablature = population[gen_best_idx]
            
            # Update overall best if needed
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                # Create a deep copy to avoid accidental modification
                best_tablature = [row[:] for row in gen_best_tablature]
                
                # Print update when we find a new best solution
                print(f"Bar {bar_idx}, Generation {generation}: New best fitness = {best_fitness:.4f}")
            
            # Print progress periodically
            if generation % max(1, self.generations // 20) == 0:
                print(f"Bar {bar_idx}, Generation {generation}: Current best fitness = {gen_best_fitness:.4f}")
                print(f"Overall best fitness = {best_fitness:.4f}")
                
                # Show fitness components
                pc = self.calculate_playability(gen_best_tablature)
                nwc = self.calculate_NWC(gen_best_tablature, positions)
                ncc = self.calculate_NCC(gen_best_tablature)
                
                print(f"PC: {pc:.4f}, NWC: {nwc:.4f}, NCC: {ncc:.4f}")
                
            # Create the next generation
            new_population = []
            
            # IMPORTANT FIX: Always include the overall best solution (true elitism)
            if best_tablature is not None:
                new_population.append([row[:] for row in best_tablature])  # Deep copy
            
            # Add elite individuals from current generation
            elite_count = max(1, int(0.1 * self.population_size)) - 1  # -1 because we already added the overall best
            for i in range(min(elite_count, len(sorted_indices))):
                # Skip if this is the same as the overall best we already added
                if i == 0 and gen_best_fitness >= best_fitness:
                    continue
                new_population.append([row[:] for row in population[sorted_indices[i]]])
            
            # Fill the rest of the population with offspring from the top individuals
            while len(new_population) < self.population_size:
                # Simply use the top two individuals as parents - keeping your existing approach
                parent1 = population[sorted_indices[0]]
                parent2 = population[sorted_indices[1]]
                
                # Apply crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self.two_point_crossover(parent1, parent2)
                else:
                    child1, child2 = [row[:] for row in parent1], [row[:] for row in parent2]  # Deep copy
                
                # Apply mutation
                child1 = self.mutate(child1, positions)
                child2 = self.mutate(child2, positions)
                
                # Add children to new population
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            # Replace old population
            population = new_population
        
        # IMPORTANT FIX: Return the best solution found across ALL generations
        # This prevents returning a worse solution than the best one found
        print(f"\nFinal result for bar {bar_idx}:")
        print(f"Best fitness found: {best_fitness:.4f}")
        
        # Calculate component values for the best tablature
        pc = self.calculate_playability(best_tablature)
        nwc = self.calculate_NWC(best_tablature, positions)
        ncc = self.calculate_NCC(best_tablature)
        
        print(f"Fitness Components:")
        print(f"Playability (PC): {pc:.4f}")
        print(f"Note Weight (NWC): {nwc:.4f}")
        print(f"Notes in Chord (NCC): {ncc:.4f}")
        
        return best_tablature
    
    def run(self):
        """Run the genetic algorithm for all bars"""
        results = []
        
        for i in range(len(self.bars_data)):
            print(f"\nProcessing bar {i+1}/{len(self.bars_data)}")
            
            best_tablature = self.run_single(i)
            results.append(best_tablature)
            
            # Display the result
            print(f"\nBest tablature found for bar {i+1}:")
            
            # Extract active positions for visualization
            active_positions = []
            for pos, chord in enumerate(best_tablature):
                if any(fret != -1 for fret in chord) and self.bars_data[i]['positions'][pos] > 0:
                    active_positions.append(chord)
            
            if active_positions:
                visualize_guitar_tab(active_positions)
            
            # Calculate component values for the best tablature
            pc = self.calculate_playability(best_tablature)
            nwc = self.calculate_NWC(best_tablature, self.bars_data[i]['positions'])
            ncc = self.calculate_NCC(best_tablature)
            
            print(f"\nFitness Components:")
            print(f"Playability (PC): {pc:.4f}")
            print(f"Note Weight (NWC): {nwc:.4f}")
            print(f"Notes in Chord (NCC): {ncc:.4f}")
            print(f"Total Fitness: {pc + nwc + ncc:.4f}")
        
        return results