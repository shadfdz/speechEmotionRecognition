import numpy as np
import librosa
import pandas as pd
import os
import librosa.display
import matplotlib.pyplot as plt

def main():
    ravdess_emotions = {'01': "neutral", "02":"calm", "03":"happy", "04":"sad", "05":"angry","06":"fearful",
                        "07":"disgust", "08":"surprised"}


    data_path = "/Volumes/Transcend/BDA600/ravdess/"
    processed_data = []
    for i in os.listdir(data_path):
        if not i.startswith("."):
            for j in os.listdir(data_path + i):
                if not j.startswith("."):
                    emotion = ravdess_emotions[j.split("-")[2]]
                    processed_data.append([data_path + i + "/" + j, j, emotion])

    processed_df = pd.DataFrame(processed_data, columns=["path","filename","emotion"])

    audio_array, sampling_array = librosa.load(processed_df["path"][0], duration=3.5, offset=0.6)
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio_array, sr= sampling_array)
    plt.show()














if __name__ == "__main__":
    main()