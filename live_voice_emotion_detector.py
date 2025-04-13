import sounddevice as sd
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import noisereduce as nr

DURATION = 4  # seconds
SAMPLE_RATE = 22050
MODEL_PATH = "voice_emotion_detector.h5"

# Load model and labels
model = load_model(MODEL_PATH)
emotions = ["sleepy", "angry", "stress"]

# Feature Extraction
def extract_features_from_audio(audio, sr):
    # Trim silence
    audio, _ = librosa.effects.trim(audio)

    # Normalize audio
    audio = librosa.util.normalize(audio)

    # Denoise audio
    audio = nr.reduce_noise(y=audio, sr=sr)

    # Extract MFCCs (n_mfcc=40)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    # Mean across time axis → (40,)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # Reshape to (1, 1, 40)
    return mfccs_mean.reshape(1, 1, 40)

#  Prediction Logic
def record_and_predict():
    print("\nRecording... Speak now!")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()

    # Ensure length is exactly 4 seconds
    expected_length = SAMPLE_RATE * DURATION
    if len(audio) < expected_length:
        padding = np.zeros(expected_length - len(audio))
        audio = np.concatenate((audio, padding))
    elif len(audio) > expected_length:
        audio = audio[:expected_length]

    # Extract features
    features = extract_features_from_audio(audio, SAMPLE_RATE)
    print("Final feature shape:", features.shape)  # Should be (1, 1, 40)

    # Predict
    print("Predicting...")
    prediction = model.predict(features)[0]

    emotion_index = np.argmax(prediction)
    emotion = emotions[emotion_index]
    confidence = prediction[emotion_index]

    print(f"Predicted Emotion: {emotion.upper()} (Confidence: {confidence:.2f})")

# Loop
try:
    while True:
        record_and_predict()
        cont = input("\nPress Enter to continue or type 'q' to quit: ")
        if cont.lower() == 'q':
            print("Exiting live emotion detector.")
            break
except KeyboardInterrupt:
    print("\nStopped by user.")
