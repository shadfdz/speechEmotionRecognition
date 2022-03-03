import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ravdess_emotions = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }

    crema_emotions = {
        "SAD": "sad",
        "ANG": "angry",
        "DIS": "disgust",
        "FEA": "fearful",
        "HAP": "happy",
        "NEU": "neutral",
    }
    savee_emotions = {
        "a": "angry",
        "d": "disgust",
        "f": "fearful",
        "h": "happy",
        "n": "neutral",
        "sa": "sad",
        "su": "surprised",
    }

    ravdess_path = "/Volumes/Transcend/BDA600/data/Ravdess/audio_speech_actors_01-24/"  # pragma: allowlist secret
    data_path = "/Volumes/Transcend/BDA600/data/"
    processed_data = []
    # loop through ravdess dataset and add path, filename, and emotion to the list
    for i in os.listdir(ravdess_path):
        if not i.startswith("."):
            for j in os.listdir(ravdess_path + i):
                if not j.startswith("."):
                    emotion = ravdess_emotions[j.split("-")[2]]
                    processed_data.append([ravdess_path + i + "/" + j, j, emotion])

    # loop through crema, savee, and tess data and add path, filename, and emotion to the list
    for i in os.listdir(data_path):
        if i == "Ravdess" or i.startswith("."):
            continue
        for j in os.listdir(data_path + i):
            if not j.startswith("."):
                if i == "Crema":
                    emotion = crema_emotions[j.split("_")[2]]
                elif i == "Tess":
                    if j.split("_")[-1].split(".")[0] == "ps":
                        emotion = "surprised"
                    else:
                        emotion = j.split("_")[-1].split(".")[0]
                else:
                    split = j.split("_")[-1]
                    if len(split) == 7:
                        emotion = savee_emotions[split[0]]
                    else:
                        emotion = savee_emotions[split[0] + split[1]]

                processed_data.append([data_path + i + "/" + j, j, emotion])

    # turn list of all data into pandas dataframe
    processed_df = pd.DataFrame(processed_data, columns=["path", "filename", "emotion"])

    audio_array, sampling_array = librosa.load(
        processed_df["path"][0], duration=3.5, offset=0.6
    )
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio_array, sr=sampling_array)
    plt.title(processed_df["emotion"][0])
    plt.show()


if __name__ == "__main__":
    main()
