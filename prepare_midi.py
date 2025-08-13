from remi_z import MultiTrack as mt
import os
midi_fp='midis/caihong.mid'
output_dir = 'caihong_clip/'
st,end = 68,76
clip = mt.from_midi(midi_fp)[st:end]
os.makedirs(output_dir,exist_ok=True)
clip.to_midi(output_dir+f'caihong_clip_{st}_{end}.mid')