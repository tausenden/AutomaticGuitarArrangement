import numpy as np
import random
import time
import json
import os
from ..utils import Guitar, pitch2name, visualize_guitar_tab, GATab, GATabSeq
from remi_z import MultiTrack

class GAreproducing:
    """
    Genetic Algorithm for Guitar Arrangement (Position-Based Approach).
    This class arranges guitar music by optimizing playability, note accuracy, and chord richness.
    """
    def __init__(self, guitar=None, 
                 mutation_rate=0.03, crossover_rate=0.6,
                 population_size=300, generations=100, 
                 num_strings=6, max_fret=20,
                 weight_PC=1.0, weight_NWC=1.0, weight_NCC=1.0,
                 midi_file_path=None, tournament_k=5, resolution=16):
        self.tournament_k = tournament_k
        """
        Initialize the genetic algorithm for guitar arrangement.
        Args:
            guitar (Guitar, optional): Guitar instance for fretboard calculations.
            mutation_rate (float): Probability of mutation per position.
            population_size (int): Number of individuals in the population.
            generations (int): Number of generations to run.
            num_strings (int): Number of guitar strings.
            max_fret (int): Maximum fret position to consider.
            midi_file_path (str): Path to the MIDI file to process.
            weight_PC (float): Weight for playability.
            weight_NWC (float): Weight for note weight component.
            weight_NCC (float): Weight for notes in chord component.
            crossover_rate (float): Probability of crossover.
            tournament_k (int): Tournament size for selection.
            resolution (int): Time resolution per bar (default 16 for 16th notes).
        """
        self.guitar = guitar if guitar is not None else Guitar()
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.generations = generations
        self.num_strings = num_strings
        self.max_fret = max_fret
        self.crossover_rate = crossover_rate
        self.weight_PC = weight_PC
        self.weight_NWC = weight_NWC
        self.weight_NCC = weight_NCC
        self.note_categories = {}
        self.category_weights = {'melody': 2.0, 'harmony': 1.0}
        self.tournament_k = tournament_k
        self.resolution = resolution

        if not midi_file_path:
            raise ValueError("MIDI file path is required")
        mt = MultiTrack.from_midi(midi_file_path)
        self.bars_data = []
        self.target_melody_list = []
        remiz_resolution = 48
        self.resolution_scale = remiz_resolution // self.resolution
        for bar in mt.bars:
            original_midi_pitches = [[] for _ in range(self.resolution)]
            original_onsets = []  # 保存实际的onset时间
            melody_notes = bar.get_melody('hi_track')
            all_notes = bar.get_all_notes(include_drum=False)
            for note in all_notes:
                note.onset = int(note.onset//self.resolution_scale)
                note.duration = int(note.duration//self.resolution_scale)
            # 收集所有onset时间
            onset_times = set()
            for note in all_notes:
                onset_times.add(int(note.onset//self.resolution_scale))
            original_onsets = sorted(list(onset_times))
            
            for note in all_notes:
                pos = note.onset
                if 0 <= pos < self.resolution:
                    original_midi_pitches[pos].append(note.pitch)
            chords = bar.get_chord()
            chord_name = []
            for chord in chords:
                if chord is None:
                    chord_name.append('C')
                else:
                    chord_name.append(chord[0]+chord[1])
            self.bars_data.append({
                'original_midi_pitches': original_midi_pitches, 
                'chords': chord_name,
                'original_onsets': original_onsets  # 添加onset时间信息
            })
            self.target_melody_list.append([note.pitch for note in melody_notes])
            # Improved melody detection: All positions with notes are considered
            self.melody_positions = set()
            for pos, pitches in enumerate(original_midi_pitches):
                if pitches:  # Any position with notes is considered melody
                    self.melody_positions.add(pos)
            
            bar_categories = {}
            for pos in range(self.resolution):
                if pos in self.melody_positions:
                    bar_categories[pos] = 'melody'
                elif original_midi_pitches[pos]:
                    bar_categories[pos] = 'harmony'
            self.note_categories.update(bar_categories)
        
        # Initialize statistics tracking
        self.statistics = {
            'processing_time_seconds': 0.0,
            'total_bars': len(self.bars_data),
            'fitness_stats': {
                'avg_PC': 0.0,
                'avg_NWC': 0.0,
                'avg_NCC': 0.0,
                'avg_fitness': 0.0
            },
            'ga_config': {
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'population_size': self.population_size,
                'generations': self.generations,
                'num_strings': self.num_strings,
                'max_fret': self.max_fret,
                'weight_PC': self.weight_PC,
                'weight_NWC': self.weight_NWC,
                'weight_NCC': self.weight_NCC,
                'tournament_k': self.tournament_k,
                'resolution': self.resolution
            }
        }

    def initialize_population(self, bar_idx):
        """
        Create a random population of tablatures for a specific bar.
        Returns:
            list: Population of tablatures (list of lists).
        """
        original_midi_pitches = self.bars_data[bar_idx]['original_midi_pitches']
        population = []
        for _ in range(self.population_size):
            tablature = []
            for pos in range(self.resolution):
                if not original_midi_pitches[pos]:
                    tablature.append([-1] * self.num_strings)
                else:
                    chord = [random.randint(0, self.max_fret) if random.random() >= 0.8 else -1 for _ in range(self.num_strings)]
                    if all(fret == -1 for fret in chord):
                        chord[random.randint(0, self.num_strings - 1)] = random.randint(0, self.max_fret)
                    tablature.append(chord)
            population.append(tablature)
        return population

    def calculate_playability(self, tablature):
        """
        Compute playability score for a tablature (higher is easier to play).
        Args:
            tablature (list): List of chord fingerings.
        Returns:
            float: Playability score.
        """
        active_positions = [pos for pos, chord in enumerate(tablature) if any(fret != -1 for fret in chord)]
        if not active_positions:
            return 0
        active_chords = [tablature[pos] for pos in active_positions]
        played_strings_penalty = -sum(sum(1 for fret in chord if fret > -1) for chord in active_chords)
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
        span_difficulty = 0
        for chord in active_chords:
            pressed = [fret for fret in chord if fret > 0]
            if not pressed:
                continue
            min_fret = min(pressed)
            max_fret = max(pressed)
            span = max_fret - min_fret + 1
            if span > 4:
                span_difficulty -= (span - 4) ** 2
        playability = (played_strings_penalty + fret_distance_penalty + span_difficulty)
        return playability

    def calculate_NWC(self, tablature, original_midi_pitches):
        """
        Compute Note Weight Component (NWC) for a tablature.
        Args:
            tablature (list): List of chord fingerings.
            original_midi_pitches (list of lists): All MIDI pitches at each position in the original MIDI.
        Returns:
            float: NWC score.
        """
        total_score = 0
        for pos, (chord, targets) in enumerate(zip(tablature, original_midi_pitches)):
            if not targets:
                continue
            # For NWC, use the highest pitch as the main target (melody)
            target = max(targets)
            chord_dict = {j+1: fret for j, fret in enumerate(chord)}
            midi_notes = [note for note in self.guitar.get_chord_midi(chord_dict) if note != -1]
            if not midi_notes:
                total_score -= 100  # Large penalty for missing the entire position
                continue
            category = self.note_categories.get(pos, 'harmony')
            weight = self.category_weights.get(category, 1.0)
            highest_note = max(midi_notes)
            pitch_distance = abs(highest_note - target)
            if pitch_distance == 0:
                total_score += 10
            else:
                total_score -= pitch_distance ** 2 * weight 
        return total_score

    def calculate_NCC(self, tablature, original_midi_pitches):
        """
        Compute Notes in Chord Component (NCC) for a tablature using the exponential curve from the paper.
        Args:
            tablature (list): List of chord fingerings.
            original_midi_pitches (list of lists): All MIDI pitches at each position in the original MIDI.
        Returns:
            float: NCC score (sum over all positions, with diminishing returns for more notes included).
        """
        total_ncc = 0
        for chord, targets in zip(tablature, original_midi_pitches):
            if not targets:
                continue  # No notes in original MIDI at this position
            chord_dict = {j+1: fret for j, fret in enumerate(chord)}
            midi_notes = set(self.guitar.get_chord_midi(chord_dict))
            midi_notes.discard(-1)
            included = sum(1 for t in targets if t in midi_notes)
            total_ncc += 1 - np.exp(-0.5 * included)
        return total_ncc

    def fitness(self, tablature, original_midi_pitches):
        """
        Calculate the overall fitness of a tablature.
        Args:
            tablature (list): List of chord fingerings.
            original_midi_pitches (list): Target note positions.
        Returns:
            float: Weighted fitness score.
        """
        pc = self.calculate_playability(tablature)
        nwc = self.calculate_NWC(tablature, original_midi_pitches)
        ncc = self.calculate_NCC(tablature, original_midi_pitches)
        return pc * self.weight_PC + nwc * self.weight_NWC + ncc * self.weight_NCC

    def two_point_crossover(self, parent1, parent2):
        """
        Perform two-point crossover between two parent tablatures.
        Args:
            parent1, parent2 (list): Parent tablatures.
        Returns:
            tuple: Two child tablatures.
        """
        if len(parent1) < 3:
            return parent1.copy(), parent2.copy()
        point1, point2 = sorted(random.sample(range(len(parent1)), 2))
        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        return child1, child2

    def mutate(self, tablature, original_midi_pitches):
        """
        Mutate a tablature by randomly changing one string in a position.
        Args:
            tablature (list): Tablature to mutate.
            original_midi_pitches (list): Target note positions.
        Returns:
            list: Mutated tablature.
        """
        for pos in range(len(tablature)):
            if random.random() < self.mutation_rate and original_midi_pitches[pos]:
                string_idx = random.randint(0, self.num_strings - 1)
                if random.random() < 0.3:
                    tablature[pos][string_idx] = -1
                else:
                    tablature[pos][string_idx] = random.randint(0, self.max_fret)
                # if all(fret == -1 for fret in tablature[pos]):
                #     tablature[pos][random.randint(0, self.num_strings - 1)] = random.randint(0, self.max_fret)
        return tablature

    def run_single(self, bar_idx):
        """
        Run the genetic algorithm for a single bar.
        Args:
            bar_idx (int): Index of the bar to process.
        Returns:
            list: Best tablature found for the bar.
        """
        bar_data = self.bars_data[bar_idx]
        original_midi_pitches = bar_data['original_midi_pitches']
        population = self.initialize_population(bar_idx)
        best_tablature = None
        best_fitness = float('-inf')
        tournament_k = self.tournament_k  # Tournament size for selection
        for generation in range(self.generations):
            fitnesses = [self.fitness(tab, original_midi_pitches) for tab in population]
            sorted_indices = np.argsort(fitnesses)[::-1]
            gen_best_idx = sorted_indices[0]
            gen_best_fitness = fitnesses[gen_best_idx]
            gen_best_tablature = population[gen_best_idx]
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_tablature = [row[:] for row in gen_best_tablature]
            if generation % max(1, self.generations // 10) == 0:
                pc = self.calculate_playability(gen_best_tablature)
                nwc = self.calculate_NWC(gen_best_tablature, original_midi_pitches)
                ncc = self.calculate_NCC(gen_best_tablature, original_midi_pitches)
                print(f"[Bar {bar_idx} | Gen {generation}] PC: {pc:.4f}, NWC: {nwc:.4f}, NCC: {ncc:.4f}, Total: {pc + nwc + ncc:.4f}")
                print("Current best tab:")
                visualize_guitar_tab(gen_best_tablature)
            # Elitism: keep the best
            # new_population = [population[gen_best_idx]]
            new_population = []
            # Fill the rest of the population using tournament selection
            while len(new_population) < self.population_size:
                # Tournament selection for parent1
                candidates1 = random.sample(population, tournament_k)
                parent1 = max(candidates1, key=lambda ind: self.fitness(ind, original_midi_pitches))
                # Tournament selection for parent2
                candidates2 = random.sample(population, tournament_k)
                parent2 = max(candidates2, key=lambda ind: self.fitness(ind, original_midi_pitches))
                # Crossover and mutation
                if random.random() < self.crossover_rate:
                    child1, child2 = self.two_point_crossover(parent1, parent2)
                else:
                    child1, child2 = [row[:] for row in parent1], [row[:] for row in parent2]
                child1 = self.mutate(child1, original_midi_pitches)
                child2 = self.mutate(child2, original_midi_pitches)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            population = new_population
        return best_tablature

    def run(self):
        """
        Run the genetic algorithm for all bars in the MIDI file.
        Returns:
            GATabSeq: Encapsulated arrangement results with onset time information.
        """
        # Start timing
        start_time = time.time()
        
        results = []
        total_pc = 0.0
        total_nwc = 0.0
        total_ncc = 0.0
        
        for bar_idx in range(len(self.bars_data)):
            best_tablature = self.run_single(bar_idx)
            
            # 创建GATab对象，保存onset时间信息
            bar_data = self.bars_data[bar_idx]
            ga_tab = GATab(
                bar_data=best_tablature,
                original_onsets=bar_data.get('original_onsets', [])
            )
            
            # 添加和弦信息
            chords = bar_data['chords']
            if len(chords) > 0:
                ga_tab.add_chord_info(0, chords[0])
            if len(chords) > 1:
                ga_tab.add_chord_info(self.resolution // 2, chords[1])
            
            for pos in self.melody_positions:
                ga_tab.add_melody_position(pos)
            
            results.append(ga_tab)
            
            # Use the structured GATab object for final fitness calculation
            pc = self.calculate_playability(ga_tab.bar_data)
            nwc = self.calculate_NWC(ga_tab.bar_data, self.bars_data[bar_idx]['original_midi_pitches'])
            ncc = self.calculate_NCC(ga_tab.bar_data, self.bars_data[bar_idx]['original_midi_pitches'])
            
            # Accumulate fitness statistics
            total_pc += pc
            total_nwc += nwc
            total_ncc += ncc
            
            print(f"[Bar {bar_idx+1}] PC: {pc:.4f}, NWC: {nwc:.4f}, NCC: {ncc:.4f}, Total: {pc + nwc + ncc:.4f}")
        
        # End timing and calculate statistics
        end_time = time.time()
        processing_time = end_time - start_time
        
        num_bars = len(self.bars_data)
        total_fitness = total_pc * self.weight_PC + total_nwc * self.weight_NWC + total_ncc * self.weight_NCC
        
        # Update statistics
        self.statistics['processing_time_seconds'] = round(processing_time, 2)
        self.statistics['fitness_stats']['avg_PC'] = round(total_pc / num_bars, 4)
        self.statistics['fitness_stats']['avg_NWC'] = round(total_nwc / num_bars, 4)
        self.statistics['fitness_stats']['avg_NCC'] = round(total_ncc / num_bars, 4)
        self.statistics['fitness_stats']['avg_fitness'] = round(total_fitness / num_bars, 4)
        
        # Return GATabSeq object - the structured approach
        ga_tab_seq = GATabSeq(results)
        return ga_tab_seq
    
    def get_statistics(self):
        """
        Get the statistics for the current run.
        Returns:
            dict: Statistics including timing and fitness information.
        """
        return self.statistics.copy()
    
    def export_statistics(self, output_dir, song_name=None):
        """
        Export statistics to a JSON file.
        Args:
            output_dir (str): Directory to save the statistics file.
            song_name (str, optional): Name of the song for the statistics.
        """
        if song_name:
            self.statistics['song_name'] = song_name
            filename = f"{song_name}_fitness.json"
        else:
            raise ValueError("Song name is None")
        
        os.makedirs(output_dir, exist_ok=True)
        statistics_file_path = os.path.join(output_dir, filename)
        
        with open(statistics_file_path, 'w') as f:
            json.dump(self.statistics, f, indent=2)
        
        return statistics_file_path