import pandas as pd
import numpy as np
from scipy import stats
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import librosa

def cont_cont_heatmap(continuous, dataframe):
    result = []
    for i in continuous:
        holder = []
        for j in continuous:
            holder.append(
                np.round(stats.pearsonr(dataframe[i].values, dataframe[j].values)[0], 3)
            )
        result.append(holder)

    fig = ff.create_annotated_heatmap(
        result, x=continuous, y=continuous, showscale=True, colorscale="Blues"
    )
    fig.update_layout(title="Continuous-Continuous Correlation Matrix")
    fig.show()

def extract_mfcc(file_name):
    audio, sample_rate = librosa.load(file_name)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=50)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    #print(len(mfccs_processed))
    return mfccs_processed

def extract_mel(file_name):
    audio, sample_rate = librosa.load(file_name)
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels = 50, power=1)
    mel_processed = np.mean(mel.T, axis=0)
    #mel_processed = np.log(mel_processed)
    print(len(mel_processed))
    return mel_processed

def extract_lpc(file_name):
    audio, sample_rate = librosa.load(file_name)
    lpc = librosa.lpc(audio,order=12)
    print(len(lpc))
    return lpc

def extract_rms(file_name):
    audio, sample_rate = librosa.load(file_name)
    rms = librosa.feature.rms(y=audio,n_rms=50 )
    rms_processed = np.mean(rms.T, axis=1)
    print(rms_processed)
    print(len(rms_processed))
    return rms_processed



def main():
    df_path ="/Volumes/Transcend/BDA600/data_models/"
    main_df = pd.read_pickle(df_path + "main_df.pkl")

    emotion_map = {"happy":0, "sad":1, "angry":2, "fearful":3, "disgust":4, "neutral":5,
                   "surprised":6, "calm":7}
    mapped = [emotion_map[i] for i in main_df["emotion"]]
    main_df["emotion_mapped"] = mapped
    # testing to see if droppiong these labels creates better results
    #main_df = main_df[(main_df["emotion_mapped"] != 6) & (main_df["emotion_mapped"] != 7)]


    # create new dataframe which gets first 40 mfcc values in audio file
    mfcc_holder = []
    mel_holder = []
    lpc_holder = []
    rms_holder = []
    count = 0
    for i in main_df["path"]:
        print(count/len(main_df)*100)
        #mfcc_holder.append(extract_mfcc(i).tolist())
        #mel_holder.append(extract_mel(i).tolist())
        #lpc_holder.append(extract_lpc(i).tolist())
        rms_holder.append(extract_rms(i).tolist())
        #print(zero_holder)
        count += 1

    # create this datafraem
    # mfcc_df = pd.DataFrame(mfcc_holder)
    # pd.to_pickle(mfcc_df, "/Volumes/Transcend/BDA600/data_models/mfcc_df")
    #
    # # rename the columns
    # column_names = []
    # curr = 1
    # for i in range(len(mfcc_df.columns)):
    #     column_names.append("mfcc" + str(curr))
    #     curr += 1
    #
    #
    # mfcc_df.columns = column_names
    # mfcc_df["zero_crossing"] = zero_holder
    # mfcc_df["class"] = main_df["emotion_mapped"].values
    # pd.to_pickle(mfcc_df, "/Volumes/Transcend/BDA600/data_models/mfcc_df")

    mfcc_df = pd.read_pickle("/Volumes/Transcend/BDA600/data_models/mfcc_df")
    mfcc_df["mean_zero"] = main_df["mean_zero"].values
    # mel_df = pd.DataFrame(mel_holder)
    # cols = []
    # cont = 1
    # for i in range(len(mel_df.columns)):
    #     cols.append("mel" + str(cont))
    #     cont += 1
    # mel_df.columns = cols
    # pd.to_pickle(mel_df,"/Volumes/Transcend/BDA600/data_models/mel_df" )

    # lpc_df = pd.DataFrame(lpc_holder)
    # cols = []
    # cont = 1
    # for i in range(len(lpc_df.columns)):
    #     cols.append("lpc" + str(cont))
    #     cont += 1
    #
    # lpc_df.columns = cols
    # pd.to_pickle(lpc_df,"/Volumes/Transcend/BDA600/data_models/lpc_df"  )

    rms_df = pd.DataFrame(rms_holder)
    cols = []
    cont = 1
    for i in range(len(rms_df.columns)):
        cols.append("rms" + str(cont))
        cont += 1

    rms_df.columns = cols
    pd.to_pickle(rms_df,"/Volumes/Transcend/BDA600/data_models/rms_df"  )

    lpc_df = pd.read_pickle("/Volumes/Transcend/BDA600/data_models/lpc_df")
    mel_df = pd.read_pickle("/Volumes/Transcend/BDA600/data_models/mel_df")

    mfcc_df = pd.concat([mfcc_df.reset_index(drop=True),mel_df.reset_index(drop=True), lpc_df.reset_index(drop=True)], axis=1)
    mfcc_df = mfcc_df.reset_index(drop=True)
    #print(mfcc_df.head().to_string())

    mfcc_df = mfcc_df[(mfcc_df["class"] != 6) & (mfcc_df["class"] != 7)& (mfcc_df["class"] != 4)& (mfcc_df["class"] != 3)]
    #print(mfcc_df.head().to_string())





    feature_df = main_df[main_df.columns[3:]]
    x_features = feature_df[feature_df.columns[:-1]]
    #feature_list = [x for x in mfcc_df.columns if x != "class"]
    #print(np.any(np.isnan(mfcc_df["mel1"])))

    # #print(feature_list[:70])
    # x_features = mfcc_df[feature_list]
    # #x_features = x_features[[x for x in x_features.columns[:14]]]
    # x_features["mean_zero"] = mfcc_df["mean_zero"].values
    # subset = list(x_features.columns[:14])
    # subset2 = list(x_features.columns[40:61])
    # subset3 = list(x_features.columns[91:])
    # x_features = x_features[subset+subset2]
    # print(x_features.columns)

    #print(x_features.head().to_string())


    #col = [x for x in x_features.columns]
    # plot correlation matrix
    #cont_cont_heatmap(col, x_features)

    # get target variable
    target = main_df["emotion_mapped"]
    #target = mfcc_df["class"]

    # standardize x_features using standard scaler
    scaler = StandardScaler()
    x_features_scaled = scaler.fit_transform(x_features)

    #print(x_features.values)
    # create a train test split
    x_train, x_test, y_train, y_test = train_test_split(x_features_scaled, target.values, test_size=0.2, random_state=42)

    # implement 5 fold cross validation
    kf = KFold(n_splits=5)

    tree = DecisionTreeClassifier()
    svc = SVC(kernel = 'linear')
    forest = RandomForestClassifier()

    acc_score = []
    acc_score_svc = []
    acc_score_forest = []
    for train_index, test_index in kf.split(x_train):
        #print(kf.split(x_train))
        curr_X_train, curr_X_test = x_train[train_index, :], x_train[test_index, :]
        curr_y_train, curr_y_test = y_train[train_index], y_train[test_index]

        print("fitting tree")
        tree.fit(curr_X_train, curr_y_train)
        print("fitting svc")
        #svc.fit(curr_X_train, curr_y_train)
        print("fitting forest")
        forest.fit(curr_X_train, curr_y_train)
        pred_values = tree.predict(curr_X_test)
        #pred_values_svc = svc.predict(curr_X_test)
        pred_values_forest = forest.predict(curr_X_test)

        acc = accuracy_score(curr_y_test, pred_values)
        #acc_svc = accuracy_score(curr_y_test, pred_values_svc)
        acc_forest = accuracy_score(curr_y_test, pred_values_forest)

        acc_score.append(acc)
        #acc_score_svc.append(acc_svc)
        acc_score_forest.append(acc_forest)

    print(acc_score)
    #print(acc_score_svc)
    print(acc_score_forest)

    forest.fit(x_train, y_train)
    feature_importances = forest.feature_importances_
    for i in range(len(feature_importances)):
        print(feature_importances[i], x_features.columns[i])
    y_pred = forest.predict(x_test)
    print("test acc: ", accuracy_score(y_test, y_pred))









if __name__ == "__main__":
    main()
