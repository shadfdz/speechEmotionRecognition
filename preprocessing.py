import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


def create_dataframe(ravdess_path, data_path):
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
                    elif j.split("_")[-1].split(".")[0] == "fear":
                        emotion = "fearful"
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
    return processed_df


def main():

    ravdess_path = "/Volumes/Transcend/BDA600/data/Ravdess/audio_speech_actors_01-24/"  # pragma: allowlist secret
    data_path = "/Volumes/Transcend/BDA600/data/"
    processed_df = create_dataframe(ravdess_path, data_path)

    print(processed_df["emotion"].value_counts())

    fig = px.histogram(processed_df, "emotion")
    fig.show()

    happy_count = 0
    sad_count = 0
    angry_count = 0
    count = 0
    for i in processed_df["emotion"]:
        if i == "happy" and happy_count < 5:
            audio_array, sampling_array = librosa.load(
                processed_df["path"][count], duration=3.5, offset=0.6
            )
            happy_count += 1
        elif i == "sad" and sad_count < 5:
            audio_array, sampling_array = librosa.load(
                processed_df["path"][count], duration=3.5, offset=0.6
            )
            sad_count += 1
        elif i == "angry" and angry_count < 5:
            audio_array, sampling_array = librosa.load(
                processed_df["path"][count], duration=3.5, offset=0.6
            )
            angry_count += 1
        else:
            count += 1
            continue
        librosa.display.waveshow(audio_array, sr=sampling_array)
        plt.title(processed_df["emotion"][count])
        plt.show()
        count += 1


if __name__ == "__main__":
    main()
