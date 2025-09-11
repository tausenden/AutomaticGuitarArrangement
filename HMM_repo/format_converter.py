import os
from core.utils.state_format import convert_file

if __name__ == "__main__":
	# Default paths within this repo
	input_path = "./possible_positions_0814.jsonl"
	output_path = "./guitar_forms_CAGED_2.json"
	convert_file(input_path, output_path)


