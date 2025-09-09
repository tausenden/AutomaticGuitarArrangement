from .GA_improved import GAimproved
from .GA_reproduction import GAreproducing


class GAreproHooked(GAreproducing):
    """
    GAreproducing with injectable strategy hooks. Hooks can reuse GAimproved methods
    via an internal helper, avoiding code duplication.
    """

    def __init__(self, *args, pc_func=None, ncc_func=None, rp_func=None, init_func=None, weight_RP=1.0, **kwargs):
        self._hook_pc = pc_func
        self._hook_ncc = ncc_func
        self._hook_rp = rp_func
        self._hook_init = init_func
        self._hook_weight_RP = weight_RP
        super().__init__(*args, **kwargs)
        self._gai_helper = None
        self._gai_helper_args = {
            'guitar': self.guitar,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'population_size': 2,
            'generations': 1,
            'num_strings': self.num_strings,
            'max_fret': self.max_fret,
            'weight_PC': self.weight_PC,
            'weight_NWC': self.weight_NWC,
            'weight_NCC': self.weight_NCC,
            'weight_RP': 1.0,
            'midi_file_path': None,
            'tournament_k': self.tournament_k,
            'resolution': self.resolution,
        }

    def _ensure_gai_helper(self):
        if self._gai_helper is None:
            try:
                self._gai_helper = GAimproved(**self._gai_helper_args)
            except Exception:
                self._gai_helper = GAimproved(
                    guitar=self.guitar,
                    mutation_rate=self.mutation_rate,
                    crossover_rate=self.crossover_rate,
                    population_size=2,
                    generations=1,
                    num_strings=self.num_strings,
                    max_fret=self.max_fret,
                    weight_PC=self.weight_PC,
                    weight_NWC=self.weight_NWC,
                    weight_NCC=self.weight_NCC,
                    weight_RP=1.0,
                    midi_file_path=None,
                    tournament_k=self.tournament_k,
                    resolution=self.resolution,
                )
        return self._gai_helper

    # Hook adapters using GAimproved methods
    def _hook_pc_from_gai(self, tablature):
        helper = self._ensure_gai_helper()
        hand_candi = helper.get_finger_assignment(tablature)
        candidate = {'tab_candi': tablature, 'hand_candi': hand_candi}
        return helper.calculate_playability(candidate)

    def _hook_ncc_from_gai(self, tablature, original_midi_pitches):
        helper = self._ensure_gai_helper()
        hand_candi = helper.get_finger_assignment(tablature)
        candidate = {'tab_candi': tablature, 'hand_candi': hand_candi}
        chord_names = self.bars_data[self.current_bar]['chords'] if hasattr(self, 'current_bar') else None
        return helper.calculate_NCC(candidate, original_midi_pitches, chord_names) if chord_names else 0.0

    def _hook_rp_from_gai(self, tablature, bar_idx):
        helper = self._ensure_gai_helper()
        if helper.rhythm_pattern is None or bar_idx >= len(helper.rhythm_pattern):
            return 0.0
        hand_candi = helper.get_finger_assignment(tablature)
        candidate = {'tab_candi': tablature, 'hand_candi': hand_candi}
        return helper.calculate_RP(candidate, bar_idx)

    def _hook_init_from_gai(self, bar_idx):
        helper = self._ensure_gai_helper()
        pop = helper.initialize_population()
        return [c['tab_candi'] for c in pop]

    # Overrides to route to hooks
    def initialize_population(self, bar_idx):
        if self._hook_init is not None:
            return self._hook_init(self, bar_idx)
        return super().initialize_population(bar_idx)

    def calculate_playability(self, tablature):
        if self._hook_pc is not None:
            return self._hook_pc(self, tablature)
        return super().calculate_playability(tablature)

    def calculate_NCC(self, tablature, original_midi_pitches):
        if self._hook_ncc is not None:
            return self._hook_ncc(self, tablature, original_midi_pitches)
        return super().calculate_NCC(tablature, original_midi_pitches)

    def fitness(self, tablature, original_midi_pitches):
        base = super().fitness(tablature, original_midi_pitches)
        if self._hook_rp is not None and hasattr(self, 'current_bar'):
            rp = self._hook_rp(self, tablature, self.current_bar)
            return base + rp * self._hook_weight_RP
        return base

    def run_single(self, bar_idx):
        self.current_bar = bar_idx
        return super().run_single(bar_idx)


class GARepro_PCOnlyHooked(GAreproHooked):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, pc_func=GAreproHooked._hook_pc_from_gai, **kwargs)


class GARepro_NCCOnlyHooked(GAreproHooked):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, ncc_func=GAreproHooked._hook_ncc_from_gai, **kwargs)


class GARepro_RPOnlyHooked(GAreproHooked):
    def __init__(self, *args, weight_RP=1.0, **kwargs):
        super().__init__(*args, rp_func=GAreproHooked._hook_rp_from_gai, weight_RP=weight_RP, **kwargs)


class GARepro_InitOnlyHooked(GAreproHooked):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, init_func=GAreproHooked._hook_init_from_gai, **kwargs)


