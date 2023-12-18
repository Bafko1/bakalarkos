import os
import torch
import torchaudio
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


base_path = "LibriVoc"  
dev_path = os.path.join(base_path, "dev-clean")  
test_path = os.path.join(base_path, "test-clean")  
train_100_path = os.path.join(base_path, "train-clean-100")  
train_360_path = os.path.join(base_path, "train-clean-360")  

dev_text_path = os.path.join(base_path, "sample2label_dev.txt")
test_text_path = os.path.join(base_path, "sample2label_test.txt")
train_text_path = os.path.join(base_path, "sample20.txt")

batch_size = 32 
num_epochs = 10 
learning_rate = 0.001 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

fake_label = 0 
real_label = 1 

class AudioDataset(Dataset):

    def load_text_file(self, file_name):

        sample2label = {}

        with open(file_name, "r") as f:

            for line in f:

                line = line.split()

                sample_id = line[0]
                label = int(line[1])

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

        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_file_path, label = self.audio_files[idx]

        waveform, sample_rate = torchaudio.load(audio_file_path)

        resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
        waveform = resampler(waveform)

        waveform = waveform.numpy()

        if self.transform:
            waveform = self.transform(waveform)

        waveform = torch.from_numpy(waveform)
        label = torch.tensor(label)

        sample = {"waveform": waveform, "label": label}
        return sample

def plot_waveform(waveform, title="Waveform"):

    plt.figure()
    plt.title(title)
    plt.plot(waveform.T)
    plt.show()

def print_metrics(true_labels, pred_labels):
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=1)
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=1)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=1)

    print("Accuracy: {:.4f}".format(accuracy))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1-score: {:.4f}".format(f1))





def collate_fn(batch):

    waveforms = [sample["waveform"] for sample in batch]
    labels = [sample["label"] for sample in batch]
    max_len = max(waveform.shape[1] for waveform in waveforms)
    padded_waveforms = torch.zeros(len(waveforms), 1, max_len)

    for i, waveform in enumerate(waveforms):
        if waveform.shape[1] > max_len:
            padded_waveforms[i, :, :max_len] = waveform[:, :max_len]
        else:
            padded_waveforms[i, :, :waveform.shape[1]] = waveform
    labels = torch.stack(labels)

    return padded_waveforms, labels

class AudioClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(AudioClassifier, self).__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True)

        self.fc = nn.Linear(64, 7)  # 7 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.permute(0, 2, 1)  # (batch_size, seq_len, channels)

        output, (hidden_state, cell_state) = self.lstm(x)  # (batch_size, seq_len, hidden_size)

        output = output[:, -1, :]  # (batch_size, hidden_size)

        output = self.fc(output)  # (batch_size, num_classes)

        return output

model = AudioClassifier()
model.to(device) 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Vytvorenie inštancie vášho datasetu
train_dataset = AudioDataset(root_dirs=[train_100_path, train_360_path], file_name=train_text_path)
# Vytvorenie DataLoaderu pre trénovací dataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


# Assuming you have a test dataset
test_dataset = AudioDataset(root_dirs=[test_path], file_name=test_text_path)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)




unique_labels = set()
for _, label in train_dataset.audio_files:
    unique_labels.add(label)

print("Unique labels in the dataset:", unique_labels)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_true_labels = []
    train_pred_labels = []

    # Tady by měla být definována proměnná train_loader
    for i, (waveforms, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(waveforms)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        train_true_labels.extend(labels.cpu().numpy())
        train_pred_labels.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    # Ukládání modelu po každé epochě
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
    }, f'model_epoch_{epoch + 1}.pth')

    train_loss /= len(train_loader)
    print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}")
    print_metrics(train_true_labels, train_pred_labels)
    
# Kontrola existujících modelů a načtení posledního
model_files = [f for f in os.listdir() if f.startswith('model_epoch_') and f.endswith('.pth')]
if model_files:
    last_epoch = max([int(f.split('_')[2].split('.')[0]) for f in model_files])
    checkpoint = torch.load(f'model_epoch_{last_epoch}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['loss']
    print(f"Loaded model from epoch {epoch} for further training.")


model.eval()
test_loss = 0.0
test_true_labels = []
test_pred_labels = []

with torch.no_grad():
    for i, (waveforms, labels) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing"):

        waveforms = waveforms.to(device)
        labels = labels.to(device)

        outputs = model(waveforms)

        loss = criterion(outputs.squeeze(), labels.long())

        test_loss += loss.item()

        test_true_labels.extend(labels.cpu().numpy())
        test_pred_labels.extend(torch.argmax(outputs, dim=1).cpu().numpy())

test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")
print_metrics(test_true_labels, test_pred_labels)
