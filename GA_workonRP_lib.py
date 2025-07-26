import numpy as np
import random
from GAutils import Guitar
from GAutils import pitch2name, visualize_guitar_tab
from GAlib import HandGuitarGA
from remi_z import MultiTrack

def extract_rhythm_importance(midi_file_path):
    
    # Load MIDI file using remi_z
    mt = MultiTrack.from_midi(midi_file_path)
    
    # Count notes at each position across all bars
    position_counts = {}
    
    for bar in mt.bars:
        # Get all notes in the bar (excluding drums)
        all_notes = bar.get_all_notes(include_drum=False)
        
        # Count notes at each onset position within the bar
        for note in all_notes:
            position = note.onset
            if position not in position_counts:
                position_counts[position] = 0
            position_counts[position] += 1
    
    # Create a list with position counts
    if position_counts:
        max_position = max(position_counts.keys())
        all_position_counts = [position_counts.get(i, 0) for i in range(max_position + 1)]
    else:
        all_position_counts = []
    
    # Calculate average note density (dynamic, adapts to different songs)
    avg_count = sum(all_position_counts) / len(all_position_counts) if all_position_counts else 0
    
    # Classify positions by importance using dynamic average
    importance = []
    for count in all_position_counts:
        if count > avg_count:
            importance.append(2)  # Important position
        elif count > 0:
            importance.append(1)  # Less important but has notes
        else:
            importance.append(0)  # No notes
    
    return importance

class RhythmGA(HandGuitarGA):
    """
    Extends the existing HandGuitarGA from GAlib.py with rhythm pattern awareness.
    Inherits all the finger assignment, comfort calculations, and original functionality.
    """
    
    def __init__(self, target_melody, target_chords=None, guitar=None, midi_file_path=None,
                 mutation_rate=0.3, population_size=1000, generations=800, 
                 tournament_size=1000, reserved_ratio=0.2, num_strings=6, 
                 max_fret=8, num_fingers=4, w_PC=0.0, w_NWC=0.0, w_NCC=1.0, w_RP=1.0):
        """
        Initialize RhythmHandGuitarGA with all original HandGuitarGA functionality plus rhythm.
        
        Parameters:
        -----------
        w_RP : float
            Weight for the rhythm pattern fitness component
        midi_file_path : str
            Path to MIDI file for rhythm importance extraction
        All other parameters are passed to the parent HandGuitarGA class
        """
        # Initialize the parent HandGuitarGA class with all its functionality
        super().__init__(
            target_melody, target_chords, guitar,
            mutation_rate, population_size, generations,
            tournament_size, reserved_ratio, num_strings,
            max_fret, num_fingers, w_PC, w_NWC, w_NCC
        )
        
        # Add rhythm pattern functionality
        self.w_RP = w_RP
        self.rhythm_importance = None
        if midi_file_path:
            self.rhythm_importance = extract_rhythm_importance(midi_file_path)
            print(f"Extracted rhythm importance pattern with {len(self.rhythm_importance)} positions")
    
    def calculate_rhythm_pattern_fitness(self, arrangement, original_importance):
        arrangement_counts = []
        
        for position in arrangement:
            # HandGuitarGA uses (fingering, finger_assignment) tuples
            fingering, _ = position
                
            # Count how many strings are played in this position
            note_count = sum(1 for fret in fingering if fret > -1)
            arrangement_counts.append(note_count)
        
        # HARD CLASSIFICATION for arrangement significance
        # Use fixed threshold: more than 3 notes played = significant position
        arrangement_importance = []
        for count in arrangement_counts:
            if count > 3:
                arrangement_importance.append(2)  # Important position (dense)
            elif count > 0:
                arrangement_importance.append(1)  # Less important but has notes
            else:
                arrangement_importance.append(0)  # No notes
        
        # Make sure we're comparing equal length sequences
        comparison_length = min(len(original_importance), len(arrangement_importance))
        original_importance_trimmed = original_importance[:comparison_length]
        arrangement_importance_trimmed = arrangement_importance[:comparison_length]
        
        # Calculate the differences and assign penalties
        total_penalty = 0
        for orig_imp, arr_imp in zip(original_importance_trimmed, arrangement_importance_trimmed):
            diff = abs(orig_imp - arr_imp)
            
            # Apply quadratic penalty for differences
            penalty = diff * diff
                
            total_penalty += penalty
        
        # Convert penalty to fitness (higher is better)
        # Normalize by the ACTUAL arrangement length to be consistent with other fitness components
        fitness = -total_penalty / len(arrangement) if arrangement else 0
        
        return fitness
    
    def fitness(self, sequence, target_melody, target_chord):
        """
        Override fitness to include rhythm pattern fitness.
        Keeps all original HandGuitarGA fitness components and adds rhythm.
        """
        # Get the original fitness from parent HandGuitarGA class
        # This includes PC, NWC, NCC with all the finger-specific calculations
        original_fitness = super().fitness(sequence, target_melody, target_chord)
        
        # Add rhythm pattern fitness if available
        rhythm_fitness = 0
        if self.rhythm_importance:
            rhythm_fitness = self.calculate_rhythm_pattern_fitness(sequence, self.rhythm_importance)
        
        # Combine original fitness with rhythm fitness
        total_fitness = original_fitness + (rhythm_fitness * self.w_RP)
        
        return total_fitness
    
    def run_single(self, target_melody, target_chord):
        """
        Override run_single to show rhythm pattern information during evolution.
        Keeps all original HandGuitarGA behavior and adds rhythm reporting.
        """
        population = self.initialize_population(target_melody)
        
        for generation in range(self.generations):
            fitnesses = [self.fitness(ind, target_melody, target_chord) for ind in population]
            best_fit = max(fitnesses)
            best_sequence = population[fitnesses.index(best_fit)]

            if generation % max(1, (self.generations // 10)) == 0:
                melody_pitch = [self.get_melody_pitch(chord_data) for chord_data in best_sequence]
                print(f"Generation {generation}: Best Fitness = {best_fit}")
                print("Best sequence melody:", pitch2name(melody_pitch))
                
                # Show all fitness components
                pc_fitness = self.cal_PC(best_sequence)
                nwc_fitness = self.cal_NWC(best_sequence, target_melody)
                ncc_fitness = self.cal_NCC(best_sequence, target_melody, target_chord)
                
                print(f"PC now: {pc_fitness * len(best_sequence):.2f}")
                print(f"NWC now: {nwc_fitness * len(best_sequence):.2f}")
                print(f"NCC now: {ncc_fitness * len(best_sequence):.2f}")
                
                # Show rhythm fitness if available
                if self.rhythm_importance:
                    rhythm_fitness = self.calculate_rhythm_pattern_fitness(best_sequence, self.rhythm_importance)
                    print(f"RP now: {rhythm_fitness * len(best_sequence):.2f}")
                    
                    # Show rhythm pattern comparison
                    arrangement_counts = [sum(1 for fret in fingering if fret > -1) for fingering, _ in best_sequence]
                    arrangement_importance = []
                    for count in arrangement_counts:
                        if count > 3:
                            arrangement_importance.append(2)
                        elif count > 0:
                            arrangement_importance.append(1)
                        else:
                            arrangement_importance.append(0)
                    
                    comparison_length = min(len(self.rhythm_importance), len(arrangement_importance))
                    if comparison_length > 0:
                        print(f"Original rhythm:    {self.rhythm_importance[:comparison_length]}")
                        print(f"Arrangement rhythm: {arrangement_importance[:comparison_length]}")
                
                # Show tablature (extract fret positions only)
                fret_positions = [fingering for fingering, _ in best_sequence]
                print("Fret positions:", fret_positions)
                visualize_guitar_tab(fret_positions)

            if generation == self.generations - 1:
                result = self.cal_NCC(best_sequence, target_melody, target_chord)
                print('Notes not in chord:', result * len(best_sequence))

            # Use parent class selection, crossover, and mutation
            candidates = self.tournament_selection(population, fitnesses)
            new_population = []
            candidate_idx = 0

            for _ in range(self.population_size - self.reserved):
                candidate_idx %= (len(candidates) - 1)
                parent1 = candidates[candidate_idx]
                parent2 = candidates[candidate_idx + 1]
                child = self.crossover(parent1, parent2)
                child = self.mutate2(child, target_melody, target_chord)  # Use parent's mutation
                new_population.append(child)
                candidate_idx += 1

            population = new_population + candidates[:self.reserved]

        final_fitnesses = [self.fitness(ind, target_melody, target_chord) for ind in population]
        best_sequence = population[np.argmax(final_fitnesses)]
        return best_sequence
        