from pyannote.audio import Pipeline
  
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

# inference on the whole file
pipeline("file.wav")

# inference on an excerpt
from pyannote.core import Segment
excerpt = Segment(start=2.0, end=5.0)

from pyannote.audio import Audio
waveform, sample_rate = Audio().crop("file.wav", excerpt)
pipeline({"waveform": waveform, "sample_rate": sample_rate})
