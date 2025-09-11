import numpy as np
import json
import os
from remi_z import MultiTrack, Bar
from core.hmm_export import export_hmm_path
from core.utils.hmm_utils import get_all_notes_from_midi, get_notes_grouped_by_bars, _flatten_bars_to_sequence
import fluidsynth


class HMMrepro:
	"""
	Guitar HMM following the paper exactly without improvements
	"""
	
	#def __init__(self, forms_files=['states/guitar_forms_single_expanded.json', 'states/guitar_forms_multi.json']):
	def __init__(self, forms_files=['./states/guitar_forms_CAGED_2.json']):
		# Guitar configuration
		self.num_strings = 6
		self.num_frets = 20
		self.num_frets = 20
		self.open_strings = [40, 45, 50, 55, 59, 64]
		self.string_names = ['E', 'A', 'D', 'G', 'B', 'E']  # low to high E to match open_strings order
		
		# Load forms from both files
		self.forms = []
		for name in forms_files:
			forms = self._load_forms(name)
			self.forms.extend(forms)
		print(f"Loaded {len(self.forms)} forms from {forms_files}")
		
		# Precompute form groups by pitch for efficiency
		self._group_forms_by_pitch()
		
		# Determine available pitch range for octave transposition
		self.min_available_pitch = min(self.forms_by_pitch.keys())
		self.max_available_pitch = max(self.forms_by_pitch.keys())
		print(f"Available pitch range: {self.min_available_pitch} - {self.max_available_pitch}")
	
	def _load_forms(self, forms_file):
		"""Load forms from JSON file"""
		with open(forms_file, 'r') as f:
			forms = json.load(f)
		return forms
	
	def _group_forms_by_pitch(self):
		"""Group forms by the pitches they can produce"""
		self.forms_by_pitch = {}
		for i, form in enumerate(self.forms):
			for pitch in form['pitches']:
				if pitch not in self.forms_by_pitch:
					self.forms_by_pitch[pitch] = []
				self.forms_by_pitch[pitch].append(i)
	
	def _transpose_pitch_to_valid_range(self, pitch):
		"""
		Transpose a pitch to the valid range by moving it up or down by octaves.
		Returns the transposed pitch and the number of octaves moved.
		"""
		if pitch in self.forms_by_pitch:
			return pitch, 0
		
		original_pitch = pitch
		octaves_moved = 0
		
		# Try moving down by octaves if pitch is too high
		while pitch > self.max_available_pitch:
			pitch -= 12
			octaves_moved -= 1
			if pitch in self.forms_by_pitch:
				print(f"Transposed pitch {original_pitch} down {-octaves_moved} octave(s) to {pitch}")
				return pitch, octaves_moved
		
		# Reset and try moving up by octaves if pitch is too low
		pitch = original_pitch
		octaves_moved = 0
		while pitch < self.min_available_pitch:
			pitch += 12
			octaves_moved += 1
			if pitch in self.forms_by_pitch:
				print(f"Transposed pitch {original_pitch} up {octaves_moved} octave(s) to {pitch}")
				return pitch, octaves_moved
		
		print(f"Warning: Could not transpose pitch {original_pitch} to valid range")
		return original_pitch, 0
	
	def initial_probability(self, form):
		"""
		Uniform initial probability since paper doesn't specify
		"""
		return 1.0 / len(self.forms)
	
	def transition_probability(self, from_form, to_form, time_interval=1.0):
		"""
		Calculate transition probability exactly following the paper's formula:
		a_ij(d_t) ∝ (1/2d_t)exp(-|I_i - I_j|/d_t) × 1/(1+I_j) × 1/(1+W_j) × 1/(1+N_j)
		"""
		# Movement along the neck |I_i - I_j|
		movement = abs(from_form['index_pos'] - to_form['index_pos'])
		
		# Time interval d_t
		dt = max(time_interval, 0.1)  # Avoid division by zero
		
		# Laplace distribution term: (1/2d_t)exp(-|I_i - I_j|/d_t)
		laplace_factor = (1.0 / (2.0 * dt)) * np.exp(-movement*2 / dt)
		
		# Difficulty factors from paper
		index_factor = 1.0 / (1.0 + to_form['index_pos'])    # 1/(1+I_j)
		width_factor = 1.0 / (1.0 + to_form['width'])        # 1/(1+W_j)  
		finger_factor = 1.0 / (1.0 + to_form['fingers'])     # 1/(1+N_j)
		
		# Combined probability as in paper
		prob = laplace_factor * index_factor * width_factor * finger_factor
		
		return prob
	
	def output_probability(self, form, target_pitch):
		"""
		Output probability - deterministic for single notes as in paper
		"""
		# Input is always a list, take first element
		target_pitch = target_pitch[0] if target_pitch else -1
		
		return 1.0 if target_pitch in form['pitches'] else 0.0
	
	def viterbi(self, pitch_sequence, time_intervals=None):
		"""
		Viterbi algorithm implementation with automatic pitch transposition
		"""
		T = len(pitch_sequence)
		N = len(self.forms)
		
		if time_intervals is None:
			time_intervals = [1.0] * (T - 1)
		
		# Find valid forms for each pitch, with automatic transposition
		valid_forms = []
		transposed_pitches = []
		for pitch in pitch_sequence:
			# Input is always a list, take first element
			original_pitch = pitch[0] if pitch else -1
			
			if original_pitch in self.forms_by_pitch:
				valid_forms.append(self.forms_by_pitch[original_pitch])
				transposed_pitches.append([original_pitch])
			else:
				# Try to transpose the pitch to a valid range
				transposed_pitch, octaves_moved = self._transpose_pitch_to_valid_range(original_pitch)
				
				if transposed_pitch in self.forms_by_pitch:
					valid_forms.append(self.forms_by_pitch[transposed_pitch])
					transposed_pitches.append([transposed_pitch])
				else:
					print(f"Failed to find valid arrangement even after transposition for pitch {original_pitch}")
					return None
		
		# Update pitch_sequence to use transposed pitches for the rest of the algorithm
		pitch_sequence = transposed_pitches
		
		# Initialize Viterbi tables
		delta = np.full((T, N), -np.inf)
		psi = np.zeros((T, N), dtype=int)
		
		# Initialize first time step
		for i in valid_forms[0]:
			initial_prob = self.initial_probability(self.forms[i])
			output_prob = self.output_probability(self.forms[i], pitch_sequence[0])
			if initial_prob > 0 and output_prob > 0:
				delta[0, i] = np.log(initial_prob * output_prob)
		
		# Forward pass
		for t in range(1, T):
			for j in valid_forms[t]:
				max_prob = -np.inf
				best_prev = -1
				
				for i in valid_forms[t-1]:
					if delta[t-1, i] > -np.inf:
						trans_prob = self.transition_probability(
							self.forms[i], 
							self.forms[j], 
							time_intervals[t-1]
						)
						
						if trans_prob > 0:
							prob = delta[t-1, i] + np.log(trans_prob)
							
							if prob > max_prob:
								max_prob = prob
								best_prev = i
				
				if max_prob > -np.inf:
					output_prob = self.output_probability(self.forms[j], pitch_sequence[t])
					if output_prob > 0:
						delta[t, j] = max_prob + np.log(output_prob)
						psi[t, j] = best_prev
		
		# Find best final state
		final_valid = [i for i in valid_forms[-1] if delta[T-1, i] > -np.inf]
		if not final_valid:
			print("No valid path found")
			return None
		
		best_final = max(final_valid, key=lambda i: delta[T-1, i])
		
		# Backtrack
		path_indices = [0] * T
		path_indices[T-1] = best_final
		
		for t in range(T-2, -1, -1):
			path_indices[t] = psi[t+1, path_indices[t+1]]
		
		# Convert to forms
		path = [self.forms[i] for i in path_indices]
		
		return path
	
	def visualize_tablature(self, path):
		"""Create tablature visualization"""
		tab_strings = []
		# Display strings from high to low (standard tab format)
		# self.string_names = ['E', 'A', 'D', 'G', 'B', 'E'] (low to high)
		# Display order should be: E(high), B, G, D, A, E(low)
		for i in range(6):
			tab_strings.append(f"{self.string_names[5-i]}|")
		
		for i, form in enumerate(path):
			if i > 0 and i % 8 == 0:
				for string_idx in range(6):
					tab_strings[string_idx] += "|"
			
			for string_idx in range(6):
				
				fret_config_key = str(5 - string_idx)
				
				# Handle fret_config - keys are strings "0" through "5"
				fret_config = form['fret_config']
				if fret_config_key in fret_config:
					fret = fret_config[fret_config_key]
				else:
					fret = -1
				
				if fret == -1:
					tab_strings[string_idx] += "--"
				else:
					tab_strings[string_idx] += f"{fret:>2}"
				
				tab_strings[string_idx] += "-"
		
		for tab_line in tab_strings:
			print(tab_line)


# Keep module-level helpers for backward compatibility imports
get_all_notes_from_midi = get_all_notes_from_midi
get_notes_grouped_by_bars = get_notes_grouped_by_bars
_flatten_bars_to_sequence = _flatten_bars_to_sequence
