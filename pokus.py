import pandas as pd
import numpy as np
from tensorflow.keras.layers import GRU
import webbrowser
import soundfile as sf
import tempfile
from tensorflow.keras.regularizers import l2
from keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall, F1Score
from tensorflow.keras.optimizers import AdamW
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from tensorflow.keras.layers import LSTM, LayerNormalization
from tensorflow.keras.regularizers import l1
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
import os
import sys

import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from IPython.display import Audio
from IPython.display import display

import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import pandas as pd

def load_iemocap_data(IEMOCAP_dir, csv_filename):

    iemocap_df = pd.read_csv(csv_filename)

    iemocap_df = iemocap_df[iemocap_df['emotion'] != 'xxx']

    iemocap_df['path'] = IEMOCAP_dir + "/" + iemocap_df['path']

    iemocap_df.rename(columns={'emotion': 'Emotions', 'path': 'Path'}, inplace=True)

    return iemocap_df[['Emotions', 'Path']]

IEMOCAP_dir = "IEMOCAP_full_release"
csv_filename = "iemocap_full_dataset.csv"  
IEMOCAP_df = load_iemocap_data(IEMOCAP_dir, csv_filename)

def load_data():
    Ravdess = "data"
    Crema = "AudioWAV"
    Tess = "zeny"
    Savee = "AudioData"

    ravdess_directory_list = os.listdir(Ravdess)
    file_emotion = []
    file_path = []
    for dir in ravdess_directory_list:
        actor = os.listdir(os.path.join(Ravdess, dir))
        for file in actor:
            part = file.split('.')[0]
            part = part.split('-')
            file_emotion.append(int(part[2]))
            file_path.append(os.path.join(Ravdess, dir, file))

    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Ravdess_df = pd.concat([emotion_df, path_df], axis=1)
    Ravdess_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)

    crema_directory_list = os.listdir(Crema)
    file_emotion = []
    file_path = []
    for file in crema_directory_list:
        file_path.append(os.path.join(Crema, file))
        part=file.split('_')
        if part[2] == 'SAD':
            file_emotion.append('sad')
        elif part[2] == 'ANG':
            file_emotion.append('angry')
        elif part[2] == 'DIS':
            file_emotion.append('disgust')
        elif part[2] == 'FEA':
            file_emotion.append('fear')
        elif part[2] == 'HAP':
            file_emotion.append('happy')
        elif part[2] == 'NEU':
            file_emotion.append('neutral')
        else:
            file_emotion.append('Unknown')
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Crema_df = pd.concat([emotion_df, path_df], axis=1)

    tess_directory_list = os.listdir(Tess)

    file_emotion = []
    file_path = []

    for dir in tess_directory_list:

        emotion = dir.split('_')[-1].lower()

        files = os.listdir(os.path.join(Tess, dir))

        for file in files:

            file_path.append(os.path.join(Tess, dir, file))
            file_emotion.append(emotion)

    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])

    Tess_df = pd.concat([emotion_df, path_df], axis=1)

    savee_directory_list = os.listdir(Savee)

    file_emotion = []
    file_path = []

    for subdir in savee_directory_list:
        subdir_path = os.path.join(Savee, subdir)

        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".wav"):  
                    file_path.append(os.path.join(subdir_path, file))

                    emotion_code = file[0]  
                    if emotion_code == 'a':
                        emotion = 'angry'
                    elif emotion_code == 'd':
                        emotion = 'disgust'
                    elif emotion_code == 'f':
                        emotion = 'fear'
                    elif emotion_code == 'h':
                        emotion = 'happy'
                    elif emotion_code == 'n':
                        emotion = 'neutral'
                    elif emotion_code == 'sa':
                        emotion = 'sad'
                    else:
                        emotion = 'surprise'

                    file_emotion.append(emotion)

    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    path_df = pd.DataFrame(file_path, columns=['Path'])
    Savee_df = pd.concat([emotion_df, path_df], axis=1)

    data_path = pd.concat([Ravdess_df, Crema_df, Tess_df, Savee_df, IEMOCAP_df], axis = 0)
    data_path.to_csv("data_path.csv",index=False)
    return data_path

def noise_f(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

def shift_f(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch_f(data, sampling_rate, pitch_factor=0.7):
    n_steps = 12 * np.log2(pitch_factor) 
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=n_steps) 

def apply_augmentation(data, sample_rate):

    original = data

    noise = noise_f(data)

    rate = 0.8 
    stretch = librosa.effects.time_stretch(data, rate=rate)

    shift = shift_f(data)

    pitch = pitch_f(data, sample_rate)

    speed_rate = np.random.uniform(0.7, 1.3)
    speed_tune = librosa.effects.time_stretch(data, rate=speed_rate)
    return [original, noise, stretch, shift, pitch, speed_tune]

def normalize(data):
    return (data - np.mean(data)) / np.std(data)


def plot_augmentation(augmented_data, sample_rate):

    titles = ["Original Audio", "Noise Injection", "Stretching", "Shifting", "Pitch"]

    for i, data in enumerate(augmented_data):
        plt.figure(figsize=(14,4))
        plt.title(titles[i])
        librosa.display.waveshow(y=data, sr=sample_rate)
        plt.show()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            sf.write(temp_wav.name, data, sample_rate)
            webbrowser.open(temp_wav.name)

def extract_features(data, sample_rate):

    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) 

    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) 

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) 

    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) 

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) 

    return result

def create_model(input_shape, num_classes, num_neurons=256, kernel_size=3, activation='relu', dropout_rate=0.6, weight_decay=1e-4):
    model = Sequential()

    model.add(Conv1D(num_neurons // 4, kernel_size=kernel_size, activation=activation, input_shape=input_shape, padding='same', kernel_initializer='he_uniform'))
    model.add(LayerNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(num_neurons // 2, kernel_size=kernel_size + 2, activation=activation, kernel_initializer='he_uniform'))
    model.add(LayerNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(num_neurons, kernel_size=kernel_size + 4, activation=activation, kernel_initializer='he_uniform'))
    model.add(LayerNormalization())
    model.add(Dropout(dropout_rate))

    model.add(LSTM(num_neurons // 2, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())

    model.add(Dense(num_neurons, activation=activation, kernel_regularizer=l1(0.01), kernel_initializer='he_uniform'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_neurons // 2, activation=activation, kernel_regularizer=l1(0.01), kernel_initializer='he_uniform'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))

    return model

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):

    precision = Precision()
    recall = Recall()
    f1_score = F1Score()
    #optimizer = SGD(learning_rate=0.01, momentum=0.9)
    optimizer = AdamW(learning_rate=0.001, weight_decay=1e-4)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', precision, recall, f1_score])

    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)

    #history = model.fit(X_train, y_train, epochs=100, batch_size=128, validation_split=0.2, callbacks=[early_stopping, lr_scheduler, rlrp])
    model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
    history = model.fit(X_train, y_train, epochs=100, batch_size=128, validation_split=0.2, callbacks=[early_stopping, lr_scheduler, rlrp, model_checkpoint])
    model.summary()

    y_pred = model.predict(X_test)

    return y_pred, history

def get_features(path, sample_rate):

    duration = librosa.get_duration(filename=path)
    print(f"Loading file: {path}")

    if duration < 0.6:
        offset_value = 0
    else:
        offset_value = 0.6

    data, sample_rate = librosa.load(path, duration=2.5, offset=offset_value, res_type='kaiser_fast')

    if len(data) == 0:
        print(f"Empty audio data for file: {path}")
        return []

    res1 = extract_features(data, sample_rate)
    result = np.array(res1)

    noise_data = noise_f(data)
    res2 = extract_features(noise_data, sample_rate)
    result = np.vstack((result, res2)) 

    new_data = stretch(data)
    data_stretch_pitch = pitch_f(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch, sample_rate)
    result = np.vstack((result, res3)) 
    data = librosa.util.normalize(data)

    return result

def load_and_augment_data(data_path, sample_rate=22050):
    X, Y = [], []
    for index, row in data_path.iterrows():
        path = row['Path']
        emotion = row['Emotions']

        augmented_data = get_features(path, sample_rate)
        for feature in augmented_data:
            X.append(feature)
            Y.append(emotion)

    return np.array(X), np.array(Y)

def prepare_data_for_training(X, Y):
    encoder = OneHotEncoder()
    Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, encoder

def main():
    sample_rate = 22050  
    data_path = load_data()

    X, Y = load_and_augment_data(data_path, sample_rate)
    X_train, X_test, y_train, y_test, encoder = prepare_data_for_training(X, Y)

    input_shape = (X_train.shape[1], 1)
    num_classes = y_train.shape[1]

    model = create_model(input_shape, num_classes)
    y_pred, history = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)

    print("Accuracy of our model on test data : " , model.evaluate(X_test,y_test)[1]*100 , "%")

if __name__ == "__main__":
    main()
