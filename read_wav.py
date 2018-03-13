import wave
from cStringIO import StringIO

# Maybe use librosa?
# https://www.analyticsvidhya.com/blog/2017/08/audio-voice-processing-deep-learning/
# import librosa


def get_data(path, frames=8059163):  # Roughly 3:00 min
    # data, sample_rate = librosa.load(path)
    wr = wave.open(path, "rb")
    out = StringIO()
    try:
        while wr.tell() <= frames:
            data = wr.readframes(1)
            if not data:
                break
            out.write(data + "\n")
    finally:
        wr.close()
        return out
