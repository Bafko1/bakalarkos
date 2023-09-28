import os
import pydot
import pydotplus
import numpy as np
import librosa
import librosa.display
import joblib
import opensmile
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import audiomentations as aa
import soundfile as sf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Conv1D, BatchNormalization, 
                                    MaxPooling1D, Flatten)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import LSTM
from tensorflow.keras.metrics import Precision, Recall, F1Score
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from glob import glob
import re

BACKGROUND_NOISES_PATH = "background_noises"

def process_audio_files(BACKGROUND_NOISES_PATH):
    # Získajte zoznam všetkých .wav súborov v priečinku
    wav_files = [f for f in os.listdir(BACKGROUND_NOISES_PATH) if f.endswith(".wav")]
    total_files = len(wav_files)

    # Prejdite všetkými súbormi v priečinku
    for index, filename in enumerate(wav_files):
        file_path = os.path.join(BACKGROUND_NOISES_PATH, filename)
        
        # Načítajte zvukový súbor s knižnicou librosa
        audio, _ = librosa.load(file_path, sr=22050)
        
        # Detekcia a odstránenie tichej časti
        intervals = librosa.effects.split(audio, top_db=20)  # top_db je hranica ticha
        audio_trimmed = np.concatenate([audio[start:end] for start, end in intervals])
        
        rms_value = librosa.feature.rms(y=audio_trimmed)[0].mean()
        if rms_value > 0.01: 
            # Uložte zvukový súbor so vzorkovacou frekvenciou 22050 Hz pomocou knižnice soundfile
            sf.write(file_path, audio_trimmed, 22050)
        
        # Zobrazenie priebehu spracovania v percentách
        progress = (index + 1) / total_files * 100
        print(f"Spracované: {progress:.2f}%")

    print("Všetky zvukové súbory boli úspešne prekonvertované na vzorkovaciu frekvenciu 22050 Hz.")


RAVDESS_PATH = "data2"
EMOTION_MAPPING = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Angry",
    4: "Fearful",
    5: "Disgust",
    6: "Calm",
    7: "Surprised"
}

CREMA_D_PATH = "AudioWAV2"
CREMA_D_EMOTION_MAPPING = {
    "ANG": 3,
    "DIS": 5,
    "FEA": 4,
    "HAP": 1,
    "NEU": 0,
    "SAD": 2
}

TESS_PATH = "zeny2"
TESS_EMOTION_MAPPING = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
    "fear": 4,
    "disgust": 5,
    "pleasant_surprise": 7
}

SAVEE_PATH = "AudioData2"
SAVEE_EMOTION_MAPPING = {
    "a": 3,  
    "d": 5,  
    "f": 4,  
    "h": 1,  
    "n": 0,  
    "sa": 2, 
    "su": 7  
}


# Vytvorím inštanciu extraktora príznakov z knižnice opensmile
smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                        feature_level=opensmile.FeatureLevel.Functionals)

def get_label_from_filename_ravdess(filename):

    parts = filename.split("-")
    if len(parts) > 2:
        emotion = int(parts[2])
        return emotion - 1
    else:
        print(f"Warning: Unexpected filename format: {filename}")

        return None

def get_label_from_filename_crema_d(filename):

    emotion_str = filename.split("_")[2]
    return CREMA_D_EMOTION_MAPPING[emotion_str]

def get_label_from_foldername_tess(foldername):

    for emotion, label in TESS_EMOTION_MAPPING.items():
        if emotion in foldername.lower():
            return label
    print(f"Warning: Unexpected foldername format: {foldername}")
    return None

def get_label_from_filename_savee(filename):

    if filename.startswith(("sa", "su")):
        emotion_str = filename[:2].lower()

    else:
        emotion_str = filename[0].lower()
    return SAVEE_EMOTION_MAPPING.get(emotion_str, None)

# Vytvorím inštanciu triedy Compose, ktorá kombinuje viaceré augmentácie
augmenter = aa.Compose([
    # Zmením rýchlosť zvuku náhodne o 10%
    #aa.SpeedChange(min_speed_change=-0.1, max_speed_change=0.1, p=0.5),
    # Zmením výšku zvuku náhodne o pol tónu
    aa.PitchShift(min_semitones=-1, max_semitones=1, p=0.5),
    # Zmením hlasitosť zvuku náhodne o 20%
    aa.Gain(min_gain_in_db=-10, max_gain_in_db=10, p=0.5),
    # Pridám náhodný šum do zvuku
    aa.AddBackgroundNoise(sounds_path="background_noises", p=0.5)
])

def augment_audio(audio, sample_rate):
    # Použijem inštanciu augmenter na transformáciu zvuku
    augmented_audio = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented_audio
# Pridám tvoje funkcie na vytvorenie, trénovanie a vyhodnocovanie modelu

def load_and_preprocess_audio(audio_file):

    audio, sample_rate = librosa.load(audio_file, res_type='kaiser_fast')

    # Použijem extraktor príznakov z knižnice opensmile na získanie eGeMAPS príznakov
    features = smile.process_signal(audio, sample_rate).values.flatten()

    audio = augment_audio(audio, sample_rate)

    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

    chromagram = librosa.feature.chroma_stft(y=audio, sr=sample_rate)

    melspectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)

    audio_features = [librosa.feature.zero_crossing_rate(audio)[0].mean(),
                      librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0].mean(),
                      librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0].mean(),
                      librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)[0].mean(),
                      librosa.feature.rms(y=audio)[0].mean()]

    combined_features = np.concatenate((np.mean(mfccs.T, axis=0), 
                                        np.mean(chromagram.T, axis=0), 
                                        np.mean(melspectrogram.T, axis=0), 
                                        features,
                                        audio_features))

#   print(f"Number of MFCC features: {np.mean(mfccs.T, axis=0).shape[0]}")
#   print(f"Number of chromagram features: {np.mean(chromagram.T, axis=0).shape[0]}")
#   print(f"Number of melspectrogram features: {np.mean(melspectrogram.T, axis=0).shape[0]}")
#   print(f"Number of eGeMAPS features: {features.shape[0]}")
#   print(f"Number of audio features: {len(audio_features)}")
#   print(f"Number of combined features: {combined_features.shape[0]}")

    return combined_features

# Vytvorím zoznamy pre dáta a štítky pomocou zoznamových komprehencií

def load_ravdess_data():
    X_ravdess = []
    y_ravdess = []
    print("Importovanie RAVDESS datasetu...")
    # Prechádzam cez všetky priečinky v RAVDESS_PATH
    for folder in tqdm(os.listdir(RAVDESS_PATH)):
        folder_path = os.path.join(RAVDESS_PATH, folder)
        
        # Overím, či je to priečinok
        if os.path.isdir(folder_path):
            # Prechádzam cez všetky súbory v priečinku
            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    audio_file = os.path.join(folder_path, file)
                    features = load_and_preprocess_audio(audio_file)
                    X_ravdess.append(features)
                    y_ravdess.append(get_label_from_filename_ravdess(file))

    return X_ravdess, y_ravdess

def load_crema_d_data():
    X_crema_d = []
    y_crema_d = []
    print("Importovanie CREMA-D datasetu...")
    for file in tqdm(os.listdir(CREMA_D_PATH)):
        if file.endswith(".wav"):
            audio_file = os.path.join(CREMA_D_PATH, file)
            features = load_and_preprocess_audio(audio_file)
            X_crema_d.append(features)
            y_crema_d.append(get_label_from_filename_crema_d(file))

    return X_crema_d, y_crema_d

def load_tess_data():
    X_tess = []
    y_tess = []
    print("Importovanie TESS datasetu...")

    # Prechádzam cez všetky priečinky v TESS_PATH
    for folder in tqdm(os.listdir(TESS_PATH)):
        folder_path = os.path.join(TESS_PATH, folder)
        
        # Overím, či je to priečinok
        if os.path.isdir(folder_path):
            # Prechádzam cez všetky súbory v priečinku
            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    audio_file = os.path.join(folder_path, file)
                    features = load_and_preprocess_audio(audio_file)
                    X_tess.append(features)
                    y_tess.append(get_label_from_foldername_tess(folder))

    return X_tess, y_tess

def load_savee_data():
    X_savee = []
    y_savee = []
    print("Importovanie SAVEE datasetu...")

    # Prechádzam cez všetky súbory v SAVEE_PATH
    for file in tqdm(os.listdir(SAVEE_PATH)):
        if file.endswith(".wav"):
            audio_file = os.path.join(SAVEE_PATH, file)
            features = load_and_preprocess_audio(audio_file)
            X_savee.append(features)
            y_savee.append(get_label_from_filename_savee(file))

    return X_savee, y_savee



def create_model(input_shape, num_classes, num_neurons=256, kernel_size=3, activation='relu', dropout_rate=0.5, weight_decay=1e-4):

    model = Sequential()

    model.add(Conv1D(num_neurons // 4, kernel_size=kernel_size, activation=activation, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(num_neurons // 2, kernel_size=kernel_size + 2, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(num_neurons, kernel_size=kernel_size + 4, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(GRU(num_neurons // 2, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())

    model.add(Dense(num_neurons, activation=activation, kernel_regularizer=l2(weight_decay)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_neurons // 2, activation=activation, kernel_regularizer=l2(weight_decay)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))

    return model

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):

    precision = Precision()
    recall = Recall()
    f1_score = F1Score()

    optimizer = AdamW(learning_rate=0.001, weight_decay=1e-4)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', precision, recall, f1_score])

    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[early_stopping, lr_scheduler])

    model.summary()
    
    model.save('my_improved_model.keras')

    y_pred = model.predict(X_test)

    plot_roc_curve(y_test, y_pred)

    plot_confusion_matrix(y_test, y_pred)
    plt.show()

def plot_roc_curve(y_true, y_pred):

    y_pred_labels = np.argmax(y_pred, axis=1)
    y_pred_bin = label_binarize(y_pred_labels, classes=[0, 1, 2, 3, 4, 5, 6, 7])
    y_true_bin = label_binarize(y_true, classes=[0 , 1 , 2 , 3 , 4 , 5 , 6 , 7])

    n_classes = y_true_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = roc_auc_score(y_true_bin[:, i], y_pred_bin[:, i])

    # Vykreslím ROC krivku pre každú triedu emócie pomocou matplotlib
    plt.figure(figsize=(10, 10))
    for i in range(n_classes):
        label = f"{EMOTION_MAPPING[i]} (AUC = {roc_auc[i]:.2f})"
        plt.plot(fpr[i], tpr[i], label=label)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Each Class')
    plt.legend()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

def preprocess_data(X, y):
    print("Rozmery pred škálovaním:", X.shape)  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    print("Rozmery po škálovaní trénovacích dát:", X_train.shape)
    X_test = scaler.transform(X_test)
    print("Rozmery po škálovaní testovacích dát:", X_test.shape)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    joblib.dump(scaler, 'scaler.pkl')

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    return X_train, X_test, y_train, y_test

def main():
    process_audio_files(BACKGROUND_NOISES_PATH)
    # Načítanie dát z rôznych datasetov
    X_ravdess, y_ravdess = load_ravdess_data()
    X_crema_d, y_crema_d = load_crema_d_data()
    X_tess, y_tess = load_tess_data()
    X_savee, y_savee = load_savee_data()

    # Zlúčenie všetkých dát a štítkov do jedného zoznamu
    X = X_ravdess + X_crema_d + X_tess + X_savee
    y = y_ravdess + y_crema_d + y_tess + y_savee

    # Konverzia na numpy polia
    X = np.array(X)
    y = np.array(y)

    # Predspracovanie dát
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Vytvorenie modelu
    input_shape = (X_train.shape[1], 1)
    num_classes = len(np.unique(y))
    model = create_model(input_shape, num_classes)

    # Trénovanie a vyhodnotenie modelu
    train_and_evaluate_model(model, X_train, y_train, X_test, y_test)

    model.save('my_improved_model.keras')

if __name__ == "__main__":
    main()
