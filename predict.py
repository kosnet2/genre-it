"""
Example of prediction. This should run upon audio capture.

"""

import numpy as np
import keras
from keras.models import load_model, predict
import librosa
import librosa.display


# Load model
model = load_model("models/parallel/weights.best.h5")

# Load song
y, sr = librosa.load("fma_small/fma_small/000002.mp3")

# Prepare data for input
spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=1024)
spect = librosa.power_to_db(spect, ref=np.max)
X_spect = np.empty((0, 640, 128))
spect = spect[:640, :]
X_spect = np.append(X_spect, [spect], axis = 0)

# Get prediction
prediction = predict(X_spect)
print(prediction)