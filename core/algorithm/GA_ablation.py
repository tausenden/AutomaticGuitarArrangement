import numpy as np
import random
from .GA_improved import GAimproved
from .GA_reproduction import GAreproducing


class GACombined(GAreproducing):
    """
    Straight combined GA: integrates GAreproducing (tablature-based) and GAimproved (candidate/hand + RP) logic.
    Use boolean flags or weights to activate specific parts.
    """

    def __init__(self, *args,
                 use_improved_init=False,
                 use_improved_pc=False,
                 use_improved_ncc=False,
                 use_rp=False,
                 weight_RP=1.0,
                 **kwargs):
        # Capture midi_file_path before parent init (parent does not store it)
        midi_file_path = kwargs.get('midi_file_path', None)
        self._config_use_improved_init = use_improved_init
        self._config_use_improved_pc = use_improved_pc
        self._config_use_improved_ncc = use_improved_ncc
        self._config_use_rp = use_rp
        self.weight_RP = weight_RP
        super().__init__(*args, **kwargs)
        # Preserve midi_file_path on self for downstream use
        self.midi_file_path = midi_file_path

        # Build GAimproved helper with same core parameters
        helper_args = {
            'guitar': self.guitar,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'population_size': self.population_size,
            'generations': self.generations,
            'num_strings': self.num_strings,
            'max_fret': self.max_fret,
            'weight_PC': self.weight_PC,
            'weight_NWC': self.weight_NWC,
            'weight_NCC': self.weight_NCC,
            'weight_RP': self.weight_RP,
            'midi_file_path': midi_file_path,
            'tournament_k': self.tournament_k,
            'resolution': self.resolution,
        }
        # GAimproved requires midi_file_path and builds its own bars_data; we only need its utilities.
        # To avoid double parsing and heavy state, we pass the same midi path and let it parse once.
        self._gai = GAimproved(**helper_args)

    # ----- Initialization -----
    def initialize_population(self, bar_idx):
        if not self._config_use_improved_init:
            return super().initialize_population(bar_idx)
        # Use GAimproved's population, but convert to tablature list
        population_candis = self._gai.initialize_population()
        tablatures = [c['tab_candi'] for c in population_candis]
        return tablatures

    # ----- Fitness sub-terms -----
    def _calc_pc(self, tab):
        if not self._config_use_improved_pc:
            return super().calculate_playability(tab)
        hand = self._gai.get_finger_assignment(tab)
        return self._gai.calculate_playability({'tab_candi': tab, 'hand_candi': hand})

    def _calc_nwc(self, tab, original_midi_pitches):
        return super().calculate_NWC(tab, original_midi_pitches)

    def _calc_ncc(self, tab, original_midi_pitches, chord_names):
        if not self._config_use_improved_ncc or not chord_names:
            return super().calculate_NCC(tab, original_midi_pitches)
        hand = self._gai.get_finger_assignment(tab)
        return self._gai.calculate_NCC({'tab_candi': tab, 'hand_candi': hand}, original_midi_pitches, chord_names)

    def _calc_rp(self, tab, bar_idx):
        if not self._config_use_rp:
            return 0.0
        hand = self._gai.get_finger_assignment(tab)
        return self._gai.calculate_RP({'tab_candi': tab, 'hand_candi': hand}, bar_idx)

    # ----- Combined fitness -----
    def fitness(self, tablature, original_midi_pitches, bar_idx=None):
        chord_names = None
        if bar_idx is not None and 0 <= bar_idx < len(self.bars_data):
            chord_names = self.bars_data[bar_idx]['chords']
        pc = self._calc_pc(tablature)
        nwc = self._calc_nwc(tablature, original_midi_pitches)
        ncc = self._calc_ncc(tablature, original_midi_pitches, chord_names)
        rp = self._calc_rp(tablature, bar_idx if bar_idx is not None else 0)
        return pc * self.weight_PC + nwc * self.weight_NWC + ncc * self.weight_NCC + rp * self.weight_RP

    # ----- GA loop (reproduction style over tablature) -----
    def run_single(self, bar_idx):
        bar_data = self.bars_data[bar_idx]
        if all(len(pitches) == 0 for pitches in bar_data['original_midi_pitches']):
            return [[-1] * self.num_strings for _ in range(self.resolution)]
        original_midi_pitches = bar_data['original_midi_pitches']
        population = self.initialize_population(bar_idx)
        best_tab = None
        best_fitness = float('-inf')
        for generation in range(self.generations):
            fitnesses = [self.fitness(tab, original_midi_pitches, bar_idx) for tab in population]
            best_idx = int(np.argmax(fitnesses))
            gen_best = population[best_idx]
            gen_best_fit = fitnesses[best_idx]
            if gen_best_fit > best_fitness:
                best_fitness = gen_best_fit
                best_tab = [row[:] for row in gen_best]
            if generation % max(1, self.generations // 10) == 0:
                # Optional debug similar to both classes
                pass
            new_population = []
            while len(new_population) < self.population_size:
                # Tournament selection
                cand1 = random.sample(population, self.tournament_k)
                parent1 = max(cand1, key=lambda ind: self.fitness(ind, original_midi_pitches, bar_idx))
                cand2 = random.sample(population, self.tournament_k)
                parent2 = max(cand2, key=lambda ind: self.fitness(ind, original_midi_pitches, bar_idx))
                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = super().two_point_crossover(parent1, parent2)
                else:
                    child1, child2 = [row[:] for row in parent1], [row[:] for row in parent2]
                # Mutation (re-use base mutate over tablature)
                child1 = super().mutate(child1, original_midi_pitches)
                child2 = super().mutate(child2, original_midi_pitches)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            population = new_population
        return best_tab

    def run(self):
        return super().run()


# Thin wrappers to keep existing import names working, configured via flags/weights
class GARepro_PCOnly(GACombined):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         use_improved_pc=True,
                         use_improved_ncc=False,
                         use_improved_init=False,
                         use_rp=False,
                         **kwargs)


class GARepro_NCCOnly(GACombined):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         use_improved_pc=False,
                         use_improved_ncc=True,
                         use_improved_init=False,
                         use_rp=False,
                         **kwargs)


class GARepro_RPOnly(GACombined):
    def __init__(self, *args, weight_RP=1.0, **kwargs):
        super().__init__(*args,
                         use_improved_pc=False,
                         use_improved_ncc=False,
                         use_improved_init=False,
                         use_rp=True,
                         weight_RP=weight_RP,
                         **kwargs)


class GARepro_InitOnly(GACombined):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         use_improved_pc=False,
                         use_improved_ncc=False,
                         use_improved_init=True,
                         use_rp=False,
                         **kwargs)


