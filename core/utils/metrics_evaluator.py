import numpy as np
import json
from remi_z import MultiTrack
from .GAutils import Guitar
import os


class MetricsEvaluator:
    """
    Evaluates arrangement quality metrics by comparing arranged MIDI with original MIDI.
    Provides accuracy metrics beyond fitness scores for comprehensive analysis.
    
    Metrics calculated:
    - note_accuracy: Percentage of original notes preserved (NWC-based)
    - rhythm_accuracy: Rhythm pattern correlation (RP-based) 
    - chord_accuracy: Chord progression preservation
    - melody_accuracy: Melody preservation (highest pitch approach, aligns with GA)
    - coverage_metrics: Pitch coverage and note density analysis
    """
    
    def __init__(self, original_midi_path, arranged_midi_path, resolution=16):
        """
        Initialize metrics evaluator.
        Args:
            original_midi_path (str): Path to original MIDI file
            arranged_midi_path (str): Path to arranged MIDI file  
            resolution (int): Time resolution per bar (default 16 for 16th notes)
        """
        self.original_midi_path = original_midi_path
        self.arranged_midi_path = arranged_midi_path
        self.resolution = resolution
        self.guitar = Guitar()
        
        # Load MIDI files
        self.original_mt = MultiTrack.from_midi(original_midi_path)
        self.arranged_mt = MultiTrack.from_midi(arranged_midi_path)
        
        # Extract data
        self.original_data = self._extract_midi_data(self.original_mt,single_track=False)
        self.arranged_data = self._extract_midi_data(self.arranged_mt,single_track=True)
        
    def _extract_midi_data(self, mt, single_track=False):
        bars_data = []
        remiz_resolution = 48
        resolution_scale = remiz_resolution / self.resolution
        
        for bar in mt.bars:
            # Initialize arrays for this bar
            midi_pitches = [[] for _ in range(self.resolution)]
            melody_pitches = [[] for _ in range(self.resolution)]
            # Get all notes in the bar (excluding drums) - same as GA classes
            all_notes = bar.get_all_notes(include_drum=False)
            if single_track:
                melody_notes = bar.get_melody('hi_note')
            else:
                melody_notes = bar.get_melody('hi_track')
            # Extract notes for this bar
            pass
            for note in all_notes:
                # Calculate position using same approach as GA classes
                pos = int(note.onset // resolution_scale)
                if 0 <= pos < self.resolution:
                    midi_pitches[pos].append(note.pitch)

            for note in melody_notes:
                pos = int(note.onset // resolution_scale)
                if 0 <= pos < self.resolution:
                    melody_pitches[pos].append(note.pitch)
            # Extract chord information - same as GA classes
            chords = bar.get_chord()
            chord_name = [chord[0]+chord[1] for chord in chords]
            
            bars_data.append({
                'midi_pitches': midi_pitches,
                'melody_pitches': melody_pitches,
                'chords': chord_name
            })
            
        return bars_data
    
    def calculate_note_accuracy(self):
        """
        Calculate Note Weight Component (NWC) accuracy.
        Measures what percentage of original notes are present in arranged version.
        """
        total_arranged_notes = 0
        total_matched_notes = 0
        
        min_bars = min(len(self.original_data), len(self.arranged_data))
        
        for bar_idx in range(min_bars):
            original_bar = self.original_data[bar_idx]['midi_pitches']
            arranged_bar = self.arranged_data[bar_idx]['midi_pitches']
            for pos in range(min(len(original_bar), len(arranged_bar))):
                original_notes = set(original_bar[pos])
                arranged_notes = set(arranged_bar[pos])
                for note in arranged_notes:
                    if note in original_notes:
                        total_matched_notes += 1
                    total_arranged_notes += 1
            pass
        
        if total_arranged_notes == 0:
            return 0.0
        result = total_matched_notes / total_arranged_notes
        return result
    
    def calculate_rhythm_accuracy(self):
        """
        Calculate Rhythm Pattern (RP) accuracy.
        Measures correlation between original and arranged rhythm patterns.
        """
        original_rhythm = []
        arranged_rhythm = []
        
        for bar_idx in range(len(self.original_data)):
            original_bar = self.original_data[bar_idx]['midi_pitches']
            arranged_bar = self.arranged_data[bar_idx]['midi_pitches']
            
            # Convert to rhythm pattern (1 if notes present, 0 if not)
            for pos in range(min(len(original_bar), len(arranged_bar))):
                original_rhythm.append(1 if original_bar[pos] else 0)
                arranged_rhythm.append(1 if arranged_bar[pos] else 0)
        
        if len(original_rhythm) == 0:
            return 0.0
        
        # Calculate correlation coefficient
        if np.std(original_rhythm) == 0 or np.std(arranged_rhythm) == 0:
            # If either is constant, use simple matching
            matches = sum(1 for o, a in zip(original_rhythm, arranged_rhythm) if o == a)
            return matches / len(original_rhythm)
        
        correlation = np.corrcoef(original_rhythm, arranged_rhythm)[0, 1]
        # Convert correlation (-1 to 1) to accuracy (0 to 1)
        return (correlation + 1) / 2
    
    def calculate_chord_accuracy(self):
        """
        Calculate chord accuracy by comparing extracted chords from arranged MIDI
        with original chord progressions.
        """
        chord_matches = 0
        total_comparisons = 0
        
        for bar_idx in range(len(self.original_data)):
            original_chords = self.original_data[bar_idx].get('chords')
            arranged_chords = self.arranged_data[bar_idx].get('chords')
            # Extract chord from arranged MIDI notes
            for pos in range(min(len(original_chords), len(arranged_chords))):
                original_chord = original_chords[pos]
                arranged_chord = arranged_chords[pos]
                if original_chord == arranged_chord:
                    chord_matches += 1
                total_comparisons += 1
        
        if total_comparisons == 0:
            return 0.0
        
        return chord_matches / total_comparisons
    
    def calculate_melody_accuracy(self):
        """
        Calculate melody accuracy using the same approach as GA fitness calculation.
        Uses highest pitch at each position as melody (aligns with GA's target = max(targets)).
        """
        total_melody_positions = 0
        melody_matches = 0
        
        for bar_idx in range(len(self.original_data)):
            original_bar = self.original_data[bar_idx]['melody_pitches']
            arranged_bar = self.arranged_data[bar_idx]['melody_pitches']
            
            for pos in range(min(len(original_bar), len(arranged_bar))):
                original_notes = original_bar[pos]
                arranged_notes = arranged_bar[pos]
                for pos in range(min(len(original_notes), len(arranged_notes))):
                    if original_notes[pos] == arranged_notes[pos]:
                        melody_matches += 1
                        total_melody_positions += 1
        
        if total_melody_positions == 0:
            return 0.0
        
        return melody_matches / total_melody_positions
    
    def calculate_coverage_metrics(self):
        """
        Calculate coverage metrics - how much of the original content is preserved.
        """
        # Pitch coverage
        original_pitches = set()
        arranged_pitches = set()
        
        for bar_data in self.original_data:
            for pos_notes in bar_data['midi_pitches']:
                original_pitches.update(pos_notes)
        
        for bar_data in self.arranged_data:
            for pos_notes in bar_data['midi_pitches']:
                arranged_pitches.update(pos_notes)
        
        pitch_coverage = len(arranged_pitches.intersection(original_pitches)) / len(original_pitches) if original_pitches else 0
        
        # Note density (notes per bar)
        original_density = sum(len([n for pos in bar['midi_pitches'] for n in pos]) for bar in self.original_data) / len(self.original_data)
        arranged_density = sum(len([n for pos in bar['midi_pitches'] for n in pos]) for bar in self.arranged_data) / len(self.arranged_data)
        
        density_ratio = arranged_density / original_density if original_density > 0 else 0
        
        return {
            'pitch_coverage': pitch_coverage,
            'original_density': original_density,
            'arranged_density': arranged_density, 
            'density_ratio': density_ratio
        }
    
    def evaluate_all_metrics(self):
        """
        Calculate all evaluation metrics.
        Returns:
            dict: Comprehensive metrics dictionary
        """    
        metrics = {
            'note_accuracy': round(self.calculate_note_accuracy(), 4),
            'rhythm_accuracy': round(self.calculate_rhythm_accuracy(), 4),
            'chord_accuracy': round(self.calculate_chord_accuracy(), 4),
            'melody_accuracy': round(self.calculate_melody_accuracy(), 4),
            #'coverage_metrics': self.calculate_coverage_metrics()
        }
        
        # Round coverage metrics
        # for key, value in metrics['coverage_metrics'].items():
        #     if isinstance(value, float):
        #         metrics['coverage_metrics'][key] = round(value, 4)
        
        return metrics
    
    def export_metrics(self, output_dir, song_name):
        """
        Export metrics to JSON file.
        Args:
            output_dir (str): Directory to save metrics
            song_name (str): Name of the song
        """
        metrics = self.evaluate_all_metrics()
        
        # Add metadata
        metrics['song_name'] = song_name
        metrics['original_midi'] = os.path.basename(self.original_midi_path)
        metrics['arranged_midi'] = os.path.basename(self.arranged_midi_path)
        
        os.makedirs(output_dir, exist_ok=True)
        metrics_file = os.path.join(output_dir, f"{song_name}_evaluation_metrics.json")
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics_file


def evaluate_arrangement(original_midi_path, arranged_midi_path, output_dir=None, song_name=None, resolution=16, export_separate=False):
    """
    Convenience function to evaluate an arrangement and optionally export metrics.
    
    Args:
        original_midi_path (str): Path to original MIDI
        arranged_midi_path (str): Path to arranged MIDI  
        output_dir (str, optional): Output directory - if provided, will merge with existing GA statistics
        song_name (str, optional): Song name for file naming
        resolution (int): Time resolution per bar
        export_separate (bool): Whether to export separate evaluation metrics JSON file
        
    Returns:
        dict: Evaluation metrics
    """
    evaluator = MetricsEvaluator(original_midi_path, arranged_midi_path, resolution)
    metrics = evaluator.evaluate_all_metrics()
    
    # If output_dir and song_name provided, merge with existing GA statistics
    if output_dir and song_name:
        stats_file = os.path.join(output_dir, f"{song_name}_statistics.json")
        
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                ga_data = json.load(f)
            
            # Create properly ordered structure: name → fitness → metrics → config
            ordered_data = {}
            
            # 1. Song name first
            if 'song_name' in ga_data:
                ordered_data['song_name'] = ga_data['song_name']
            
            # 2. Processing info
            if 'processing_time_seconds' in ga_data:
                ordered_data['processing_time_seconds'] = ga_data['processing_time_seconds']
            if 'total_bars' in ga_data:
                ordered_data['total_bars'] = ga_data['total_bars']
                
            # 3. Fitness stats - reorganize to proper order with only averages
            if 'fitness_stats' in ga_data:
                fitness_data = ga_data['fitness_stats']
                ordered_fitness = {}
                
                # Only keep averages in proper order: PC → NWC → NCC → RP → final fitness
                if 'avg_PC' in fitness_data:
                    ordered_fitness['avg_PC'] = fitness_data['avg_PC']
                if 'avg_NWC' in fitness_data:
                    ordered_fitness['avg_NWC'] = fitness_data['avg_NWC']
                if 'avg_NCC' in fitness_data:
                    ordered_fitness['avg_NCC'] = fitness_data['avg_NCC']
                if 'avg_RP' in fitness_data:
                    ordered_fitness['avg_RP'] = fitness_data['avg_RP']
                if 'avg_fitness' in fitness_data:
                    ordered_fitness['avg_fitness'] = fitness_data['avg_fitness']
                
                ordered_data['fitness_stats'] = ordered_fitness
            
            # 4. Evaluation metrics
            ordered_data.update(metrics)
            
            # 5. GA config last
            if 'ga_config' in ga_data:
                ordered_data['ga_config'] = ga_data['ga_config']
            
            with open(stats_file, 'w') as f:
                json.dump(ordered_data, f, indent=2)
        else:
            # Create new file with just evaluation metrics
            with open(stats_file, 'w') as f:
                json.dump(metrics, f, indent=2)
    
    # Only export separate metrics file if explicitly requested
    if export_separate and output_dir and song_name:
        evaluator.export_metrics(output_dir, song_name)
    
    return metrics
