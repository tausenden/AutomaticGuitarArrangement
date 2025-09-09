import os
import subprocess
import contextlib
from typing import Dict
from pydub import AudioSegment
from .GAutils import Guitar, GATabSeq


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def midi_to_wav(midi_fp: str, sf_path: str, wav_fp: str, verbose: bool = True) -> None:
    """Convert MIDI to WAV using fluidsynth."""
    cmd = [
        'fluidsynth',
        '-ni',
        sf_path,
        midi_fp,
        '-F', wav_fp,
        '-r', '44100',
    ]
    
    if verbose:
        subprocess.run(cmd, check=True)
    else:
        # Suppress output from fluidsynth
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def post_process_wav(wav_fp: str, silence_thresh_db: float = -40.0, padding_ms: int = 200, target_dbfs: float = -0.1) -> None:
    """Post-process WAV file by normalizing, trimming silence, and adding padding."""
    audio = AudioSegment.from_wav(wav_fp)
    change_in_dBFS = target_dbfs - audio.max_dBFS
    normalized = audio.apply_gain(change_in_dBFS)
    
    # Trim silence
    start_trim = _detect_leading_silence(normalized, silence_thresh_db)
    end_trim = _detect_leading_silence(normalized.reverse(), silence_thresh_db)
    duration = len(normalized)
    trimmed = normalized[start_trim:duration - end_trim + padding_ms]
    trimmed.export(wav_fp, format='wav')


def _detect_leading_silence(sound: AudioSegment, silence_threshold: float = -40.0, chunk_size: int = 10) -> int:
    """Detect leading silence in audio."""
    trim_ms = 0
    while trim_ms < len(sound) and sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold:
        trim_ms += chunk_size
    return trim_ms


def export_ga_results(ga_tab_seq: GATabSeq, resolution: int, song_name: str = 'you_didnt_assign_name', tempo: int = 120,
                     output_dir: str = 'outputs', sf2_path: str = 'resources/Tyros Nylon.sf2', verbose: bool = True) -> Dict[str, str]:
    """
    End-to-end export for GA arrangement results following rule_based.py and hmm_export.py pattern.
    
    Args:
        ga_tab_seq: GATabSeq object from ga_reproduction.py
        song_name: Base name for output files
        tempo: Tempo for the MIDI/audio output
        output_dir: Directory to save outputs
        sf2_path: Path to SoundFont file for audio rendering
        verbose: Whether to print progress messages
    
    Returns:
        Dict with paths to generated files: {'tab_fp', 'midi_fp', 'wav_fp'}
    """
    ensure_dir(output_dir)
    
    guitar = Guitar()
    
    # Save the tab to file
    tab_fp = os.path.join(output_dir, f'{song_name}.tab')
    if verbose:
        print(f'Saving tab to {tab_fp}')
    ga_tab_seq.save_to_file(tab_fp)
    
    # Convert to note sequence
    mt = ga_tab_seq.convert_to_multitrack(guitar, tempo=tempo,resolution=resolution)
    
    # Save the note sequence to MIDI
    midi_fp = os.path.join(output_dir, f'{song_name}.midi')
    if verbose:
        print(f'Saving MIDI to {midi_fp}')
        mt.to_midi(midi_fp)
    else:
        # Suppress REMI-z output
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                mt.to_midi(midi_fp)
    
    # Synthesize the MIDI to WAV
    wav_fp = os.path.join(output_dir, f'{song_name}.wav')
    if verbose:
        print(f'Synthesizing MIDI to WAV: {wav_fp}')
    midi_to_wav(midi_fp, sf2_path, wav_fp, verbose=verbose)
    post_process_wav(wav_fp, silence_thresh_db=-40.0, padding_ms=200, target_dbfs=-0.1)
    
    return {
        'tab_fp': tab_fp,
        'midi_fp': midi_fp,
        'wav_fp': wav_fp,
    }
