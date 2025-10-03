import numpy, scipy, librosa
y, sr = librosa.load(librosa.example('trumpet'))
print(len(y), sr)

