import librosa
import os
import torch
import torchaudio
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_auc_score, matthews_corrcoef

# Paths
base_path = "LibriVoc"
train_100_path = os.path.join(base_path, "train-clean-100")
train_360_path = os.path.join(base_path, "train-clean-360")
dev_path = os.path.join(base_path, "dev-clean")

# Txt
train_text_path = os.path.join(base_path, "sample2label_train.txt")
dev_text_path = os.path.join(base_path, "sample2label_dev.txt")

# Hyperparameters
batch_size = 14
num_epochs = 30
learning_rate = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Labels
fake_label = 0
real_label = 1

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()


class AudioDataset(Dataset):
    def load_text_file(self, file_name):
        sample2label = {}
        with open(file_name, "r") as f:
            for line in f:
                line = line.split()
                sample_id = line[0]
                label = int(line[1])
                label = fake_label if label != 0 else real_label
                sample2label[sample_id] = label
        return sample2label


    def __init__(self, root_dirs, file_name, transform=None):
        self.transform = transform
        self.audio_files = []
        self.sampling_rate = 16000

        sample2label = self.load_text_file(file_name)
        print(f"Loaded {len(sample2label)} samples from {file_name}")

        for root_dir in root_dirs:
            for sub_dir in os.listdir(root_dir):
                sub_dir_path = os.path.join(root_dir, sub_dir)
                for inner_sub_dir in os.listdir(sub_dir_path):
                    inner_sub_dir_path = os.path.join(sub_dir_path, inner_sub_dir)
                    for audio_file in os.listdir(inner_sub_dir_path):
                        audio_file_path = os.path.join(inner_sub_dir_path, audio_file)
                        sample_id = os.path.splitext(audio_file)[0]
                        label = sample2label.get(sample_id, None)
                        if label is not None:
                            self.audio_files.append((audio_file_path, label))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file_path, label = self.audio_files[idx]

        waveform, sample_rate = torchaudio.load(audio_file_path)
        resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
        waveform = resampler(waveform)

        waveform_np = waveform.numpy()[0]  # Assuming single channel audio

        # Data Augmentation (example: adding random noise)
        if self.transform:
            noise_factor = 0.005 * np.random.uniform()
            waveform_np += noise_factor * np.random.normal(size=waveform_np.shape)

        # Extrahování audio vlastností
        mfccs = librosa.feature.mfcc(y=waveform_np, sr=self.sampling_rate, n_mfcc=13)
        mfcc_length = mfccs.shape[1]

        spectral_centroids = librosa.feature.spectral_centroid(y=waveform_np, sr=self.sampling_rate)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(waveform_np)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=waveform_np, sr=self.sampling_rate)
        chroma = librosa.feature.chroma_stft(y=waveform_np, sr=self.sampling_rate)

        # Oříznutí nebo rozšíření vlastností, aby měly stejnou délku jako MFCC
        spectral_centroids = spectral_centroids[:, :mfcc_length]
        zero_crossing_rate = zero_crossing_rate[:, :mfcc_length]
        spectral_bandwidth = spectral_bandwidth[:, :mfcc_length]
        chroma = chroma[:, :mfcc_length]

        # Stack all features
        features = np.vstack((mfccs, spectral_centroids, zero_crossing_rate, spectral_bandwidth, chroma))

        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return features_tensor, label_tensor



def plot_waveform(waveform, title="Waveform"):

    plt.figure()
    plt.title(title)
    plt.plot(waveform.T)
    plt.show()

def print_metrics(true_labels, pred_labels, pred_probs=None):
    accuracy_value = accuracy_score(true_labels, pred_labels)
    precision_value = precision_score(true_labels, pred_labels, average='weighted', zero_division=1)
    recall_value = recall_score(true_labels, pred_labels, average='weighted', zero_division=1)
    f1_value = f1_score(true_labels, pred_labels, average='weighted')
    mcc_value = matthews_corrcoef(true_labels, pred_labels)

    print(f"Accuracy: {accuracy_value:.4f}")
    print(f"Precision: {precision_value:.4f}")
    print(f"Recall: {recall_value:.4f}")
    print(f"F1-score: {f1_value:.4f}")
    print(f"Matthews Correlation Coefficient: {mcc_value:.4f}")

    if pred_probs is not None:
        roc_auc_value = roc_auc_score(true_labels, pred_probs)
        print(f"ROC AUC Score: {roc_auc_value:.4f}")

    cm = confusion_matrix(true_labels, pred_labels)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def collate_fn(batch):
    # Extracting waveforms and labels from the batch
    waveforms = [item[0] for item in batch]  # item[0] is the waveform
    labels = [item[1] for item in batch]     # item[1] is the label

    # Find the length of the longest sequence in the batch
    max_len = max(waveform.size(-1) for waveform in waveforms)

    # Pad the waveforms to the length of the longest sequence
    padded_waveforms = torch.nn.utils.rnn.pad_sequence(
        [torch.nn.functional.pad(w, (0, max_len - w.size(1))) for w in waveforms],
        batch_first=True
    )

    labels = torch.stack(labels)

    return padded_waveforms, labels

def evaluate_model(model, dataloader, device):
    model.eval()
    true_labels = []
    pred_labels = []
    pred_probs = []

    with torch.no_grad():
        for waveforms, labels in dataloader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            outputs = model(waveforms)
            _, predicted = torch.max(outputs.data, 1)
            probabilities = torch.softmax(outputs, dim=1)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())
            pred_probs.extend(probabilities[:, 1].cpu().numpy()) # assuming the positive class is labeled as 1

    return true_labels, pred_labels, pred_probs

class Attention(nn.Module):
    def __init__(self, in_features, out_features):
        super(Attention, self).__init__()
        self.attention = nn.Linear(in_features, out_features)

    def forward(self, x):
        attention_weights = F.softmax(self.attention(x), dim=1)
        return attention_weights * x

class AudioClassifier(nn.Module):
    def __init__(self, num_classes=2, input_channels=28):  # Zde změňte počet vstupních kanálů na 28
        super(AudioClassifier, self).__init__()

        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(16)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.batch_norm4 = nn.BatchNorm1d(128)

        self.maxpool = nn.MaxPool1d(kernel_size=2)

        self.gru = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, dropout=0.5)

        self.fc = nn.Linear(64, num_classes)
        self.attention = Attention(128, 128)  # Initialize attention mechanism here

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.permute(0, 2, 1)  # (batch_size, seq_len, channels)

        output, _ = self.gru(x)  # (batch_size, seq_len, hidden_size)

        output = output[:, -1, :]  # (batch_size, hidden_size)

        output = self.fc(output)  # (batch_size, num_classes)

        return output

# TensorBoard setup
writer = SummaryWriter('runs/audio_classifier_experiment')

model = AudioClassifier(num_classes=2, input_channels=28)
model.to(device)  # Move the model to the GPU if available


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Instantiate datasets and dataloaders with data augmentation
train_dataset = AudioDataset(root_dirs=[train_100_path, train_360_path], file_name=train_text_path, transform=True)
dev_dataset = AudioDataset(root_dirs=[dev_path], file_name=dev_text_path, transform=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# Training Loop with Validation
# Early Stopping parameters
no_improve_epochs = 0
no_improve_threshold = 5  # Set your threshold here
best_dev_accuracy = 0.0

# Trénovací smyčka s indikátorem průběhu
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_true_labels, train_pred_labels = [], []

    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}, Training") as t:
        for waveforms, labels in t:
            waveforms, labels = waveforms.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_true_labels.extend(labels.cpu().numpy())
            train_pred_labels.extend(torch.argmax(outputs, dim=1).cpu().numpy())

            # Aktualizace průběhového indikátoru s průměrnou ztrátou
            t.set_postfix(train_loss=train_loss/len(train_loader))

    train_loss /= len(train_loader)
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")
    print_metrics(train_true_labels, train_pred_labels)
    writer.add_scalar('Training Loss', train_loss, epoch)

    # Sledování distribuce vah a gradientů
    for name, weight in model.named_parameters():
        writer.add_histogram(name, weight, epoch)
        if weight.grad is not None:
            writer.add_histogram(f'{name}.grad', weight.grad, epoch)


    # Validation on development set
    model.eval()
    dev_loss = 0.0
    dev_true_labels, dev_pred_labels = [], []
    with torch.no_grad():
        for waveforms, labels in tqdm(dev_loader, desc=f"Epoch {epoch+1}/{num_epochs}, Development"):
            waveforms, labels = waveforms.to(device), labels.to(device)
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            dev_loss += loss.item()
            dev_true_labels.extend(labels.cpu().numpy())
            dev_pred_labels.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    dev_loss /= len(dev_loader)
    dev_accuracy = accuracy_score(dev_true_labels, dev_pred_labels)
    print(f"Epoch {epoch + 1}, Development Loss: {dev_loss:.4f}, Accuracy: {dev_accuracy:.4f}")

    writer.add_scalar('Validation Loss', dev_loss, epoch)
    writer.add_scalar('Validation Accuracy', dev_accuracy, epoch)

    # Vypočítať a vypísať metriky pre trénovaciu sadu
    train_true, train_pred, train_probs = evaluate_model(model, train_loader, device)
    print(f"Training metrics for epoch {epoch+1}:")
    print_metrics(train_true, train_pred, train_probs)
    
    # Vypočítať a vypísať metriky pre validačnú (vývojovú) sadu
    dev_true, dev_pred, dev_probs = evaluate_model(model, dev_loader, device)
    print(f"Development metrics for epoch {epoch+1}:")
    print_metrics(dev_true, dev_pred, dev_probs)
    
    # Uloženie modelu, ak je presnosť na validačnej sade vyššia ako doteraz najlepšia
    if dev_accuracy > best_dev_accuracy:
        best_dev_accuracy = dev_accuracy
        # Uloženie stavu modelu
        torch.save(model.state_dict(), 'best_model.pth')
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
    
    # Skoré zastavenie, ak nedošlo k zlepšeniu
    if no_improve_epochs >= no_improve_threshold:
        print("Stopping early due to no improvement in development accuracy.")
        break
    
    # Úprava rýchlosti učenia
    scheduler.step(dev_loss)

# Načítanie najlepšieho modelu a zatvorenie TensorBoard
model.load_state_dict(torch.load('best_model.pth'))
writer.close()
