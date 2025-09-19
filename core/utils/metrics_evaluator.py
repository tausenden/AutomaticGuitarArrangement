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
        self.original_data = self._extract_midi_data(self.original_mt)
        self.arranged_data = self._extract_midi_data(self.arranged_mt)

    def _extract_midi_data(self, mt):
        bars_data = []
        remiz_resolution = 48
        resolution_scale = remiz_resolution / self.resolution
        for bar in mt.bars:
            # Initialize arrays for this bar
            midi_pitches = [[] for _ in range(self.resolution)]
            melody_pitches = [[] for _ in range(self.resolution)]
            # Get all notes in the bar (excluding drums) - same as GA classes
            try:
                all_notes = bar.get_all_notes(include_drum=False)
            except Exception:
                all_notes = []

            # Ignore empty bars (prevent melody extraction errors on leading silence)
            if len(bar) == 0 or not all_notes:
                continue

            # Safely get melody notes; fall back to empty if unavailable
            try:
                melody_notes = bar.get_melody('hi_track')
            except Exception:
                melody_notes = []
            # Extract notes for this bar
            
            for note in all_notes:
                # Calculate position using same approach as GA classes
                pos = int(note.onset // resolution_scale)
                if 0 <= pos < self.resolution:
                    midi_pitches[pos].append(note.pitch)

            for note in melody_notes:
                pos = int(note.onset // resolution_scale)
                if 0 <= pos < self.resolution:
                    melody_pitches[pos].append(note.pitch)
                    if len(melody_pitches[pos]) > 1:
                        melody_pitches[pos]=[max(melody_pitches[pos])]
            # Extract chord information - same as GA classes
            chords = bar.get_chord()
            # Filter out None values and handle cases where chord components might be None
            chord_name = []
            if chords:
                for chord in chords:
                    if chord and len(chord) >= 2 and chord[0] is not None and chord[1] is not None:
                        chord_name.append(chord[0] + chord[1])
            # If no valid chords found, use empty list
            if not chord_name:
                chord_name = []
            
            bars_data.append({
                'midi_pitches': midi_pitches,
                'melody_pitches': melody_pitches,
                'chords': chord_name
            })
            
        return bars_data
    
    def calculate_note_precision(self):
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
        """
        original_rp = []
        arranged_rp = []
        rhythm_matches = 0
        total_rhythm_positions = 0
        
        for bar_idx in range(len(self.original_data)):
            original_bar = self.original_data[bar_idx]['midi_pitches']
            arranged_bar = self.arranged_data[bar_idx]['midi_pitches']
            original_bar_rhythm=[]
            arranged_bar_rhythm=[]
            # Convert to rhythm pattern (1 if notes present, 0 if not)
            if len(original_bar) != len(arranged_bar):
                raise ValueError("original and arranged bar length is not equal")
            for pos in range(len(original_bar)):
                original_bar_rhythm.append(len(original_bar[pos]) if original_bar[pos] else 0)
                arranged_bar_rhythm.append(len(arranged_bar[pos]) if arranged_bar[pos] else 0)
            
            sum_original=sum(original_bar_rhythm)
            sum_arranged=sum(arranged_bar_rhythm)
            nonzero_o = [n for n in original_bar_rhythm if n > 0]
            nonzero_a = [n for n in arranged_bar_rhythm if n > 0]
            avg_original = sum_original/len(nonzero_o) if len(nonzero_o) > 0 else 0
            avg_arranged = sum_arranged/len(nonzero_a) if len(nonzero_a) > 0 else 0
            for pos in range(len(original_bar_rhythm)):
                if original_bar_rhythm[pos] > avg_original:
                    original_rp.append(2)
                elif original_bar_rhythm[pos] > 0:
                    original_rp.append(1)
                else:
                    original_rp.append(0)
            for pos in range(len(arranged_bar_rhythm)):
                if arranged_bar_rhythm[pos] > avg_arranged:
                    arranged_rp.append(2)
                elif arranged_bar_rhythm[pos] > 0:
                    arranged_rp.append(1)
                else:
                    arranged_rp.append(0)

        for pos in range(len(original_rp)):
            if original_rp[pos] == arranged_rp[pos]:
                rhythm_matches += 1
            total_rhythm_positions += 1

        result = rhythm_matches / total_rhythm_positions
        return result
    
    
    def calculate_chord_accuracy(self):
        """
        Calculate chord accuracy by comparing extracted chords from arranged MIDI
        with original chord progressions.
        """
        chord_matches = 0
        c_comp = 0
        cname_matches = 0
        cname_comp = 0
        
        min_bars = min(len(self.original_data), len(self.arranged_data))
        
        for bar_idx in range(min_bars):
            original_chords = self.original_data[bar_idx].get('chords', [])
            arranged_chords = self.arranged_data[bar_idx].get('chords', [])
            o_chords_name = [c[0] for c in original_chords[0]]
            a_chords_name = [c[0] for c in arranged_chords[0]]
            # Compare chords within each bar
            for pos in range(min(len(original_chords), len(arranged_chords))):
                original_chord = original_chords[pos]
                arranged_chord = arranged_chords[pos]
                if original_chord == arranged_chord:
                    chord_matches += 1
                c_comp += 1
                if original_chord[0] == arranged_chord[0]:
                    cname_matches += 1
                cname_comp += 1
            
            # Count mismatches when one bar has more chords than the other
            c_comp += abs(len(original_chords) - len(arranged_chords))
            cname_comp += abs(len(original_chords) - len(arranged_chords))
        if c_comp == 0:
            c_result = 0.0
        if cname_comp == 0:
            cname_result = 0.0
        c_result = chord_matches / c_comp
        cname_result = cname_matches / cname_comp
        return c_result, cname_result
    
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
                if len(original_bar[pos]) > 1 or len(arranged_bar[pos]) > 1:
                    raise ValueError("melody should be only one note, now multiple notes")
                if original_bar[pos]:
                    if not arranged_bar[pos]:
                        total_melody_positions += 1
                        continue
                    if original_bar[pos][0] == arranged_bar[pos][0]:
                        melody_matches += 1
                    total_melody_positions += 1
            pass
        if total_melody_positions == 0:
            return 0.0
        result = melody_matches / total_melody_positions
        return result
    
    def calculate_melody_cor(self):
        original_melody_whole=[]
        arranged_melody_whole=[]
        for bar_idx in range(len(self.original_data)):
            original_bar = self.original_data[bar_idx]['melody_pitches']
            arranged_bar = self.arranged_data[bar_idx]['melody_pitches']
            
            for pos in range(min(len(original_bar), len(arranged_bar))):
                if len(original_bar[pos]) > 1 or len(arranged_bar[pos]) > 1:
                    raise ValueError("melody should be only one note, now multiple notes")
                original_melody_whole.append(original_bar[pos][0] if original_bar[pos] else np.nan)
                arranged_melody_whole.append(arranged_bar[pos][0] if arranged_bar[pos] else np.nan)
            pass
        a = np.array(original_melody_whole, dtype=float)
        b = np.array(arranged_melody_whole, dtype=float)

        mask = ~np.isnan(a) & ~np.isnan(b)
        if mask.sum() < 2:
            return 0.0 
        result = float(np.corrcoef(a[mask], b[mask])[0, 1])
        return result

    
    def evaluate_all_metrics(self):
        """
        Calculate all evaluation metrics.
        Returns:
            dict: Comprehensive metrics dictionary
        """    
        metrics = {
            'note_precision': round(self.calculate_note_precision(), 4),
            'rhythm_accuracy': round(self.calculate_rhythm_accuracy(), 4),
            'chord_accuracy': round(self.calculate_chord_accuracy()[0], 4),
            'chord_name_accuracy': round(self.calculate_chord_accuracy()[1], 4),
            'melody_accuracy': round(self.calculate_melody_accuracy(), 4),
            'melody_correlation': round(self.calculate_melody_cor(), 4)
        }
        
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
    
    # If output_dir and song_name provided, write evaluation metrics only
    if output_dir and song_name:
        stats_file = os.path.join(output_dir, f"{song_name}_statistics.json")
        with open(stats_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    # Only export separate metrics file if explicitly requested
    if export_separate and output_dir and song_name:
        evaluator.export_metrics(output_dir, song_name)
    
    return metrics
