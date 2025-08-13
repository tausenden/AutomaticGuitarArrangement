from debug_newGAlib import GAimproved
from GAutils import Guitar, set_random

set_random(42)

ga_config={
    'guitar':Guitar(),
    'midi_file_path':'midis/caihong-4bar.midi'
}
ga = GAimproved(**ga_config)
ga.run()