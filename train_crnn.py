import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np


TRAIN_DIR = r'dataset\ukr_plates\train\images'
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Індекс 0 зарезервовано під 'blank' для CTC Loss
CHARS = "-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Розмір картинки (висота фіксована, ширина видовжена для сканування зліва направо)
IMG_HEIGHT = 64
IMG_WIDTH = 256

transform = T.Compose([
    T.Resize((IMG_HEIGHT, IMG_WIDTH)),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])  # Спрощена нормалізація
])


# --- 1. ПІДГОТОВКА ДАТАСЕТУ ---
class CRNNDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label_str = os.path.splitext(img_name)[0].upper()
        # Залишаємо тільки валідні символи, 'blank' (індекс 0) додаватиме CTC автоматично
        label_str = ''.join(c for c in label_str if c in CHARS[1:])
        label_idx = [CHAR2IDX[c] for c in label_str]

        return image, torch.tensor(label_idx, dtype=torch.long), len(label_idx)


# Спеціальна функція для об'єднання міток різної довжини в батчі
def collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images, 0)

    # Зшиваємо всі мітки в один довгий 1D тензор
    labels = torch.cat(labels, 0)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)

    return images, labels, label_lengths


# --- 2. АРХІТЕКТУРА МЕРЕЖІ (CRNN) ---
class CRNN(nn.Module):
    def __init__(self, img_channels=3, num_classes=NUM_CLASSES):
        super(CRNN, self).__init__()

        # CNN (Згорткова частина)
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x64
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # 8x64 (стискаємо висоту сильніше, ніж ширину)
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 1)),  # 4x64
            nn.Conv2d(512, 512, kernel_size=2, padding=0), nn.ReLU(),  # 3x63
        )

        # RNN (Рекурентна частина - BiLSTM)
        # Висота після CNN = 3, канали = 512. Розгортаємо їх: 3 * 512 = 1536
        self.rnn = nn.Sequential(
            nn.LSTM(1536, 256, bidirectional=True, batch_first=True),
        )

        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x)  # Розмірність: (Batch, Channels, Height, Width)
        batch, channels, height, width = conv.size()

        # Перебудовуємо тензор: (Batch, Width, Channels * Height)
        # Width стає нашою послідовністю (Time Steps)
        conv = conv.permute(0, 3, 1, 2).contiguous()
        conv = conv.view(batch, width, channels * height)

        rnn_out, _ = self.rnn(conv)  # rnn_out: (Batch, Width, 512)

        # Вихід: (Batch, TimeSteps, NumClasses)
        output = self.fc(rnn_out)

        output = output.permute(1, 0, 2)

        # CTC Loss очікує log_softmax
        return torch.nn.functional.log_softmax(output, dim=2)


# --- 3. ЦИКЛ НАВЧАННЯ ---
def train_model():
    print("Ініціалізація датасету для CRNN...")
    train_dataset = CRNNDataset(TRAIN_DIR, transform=transform)

    # Використовуємо наш кастомний collate_fn
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = CRNN().to(device)

    # blank=0 - це наш символ "-" на початку словника
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nПочинаємо навчання CRNN...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels, target_lengths in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()

            # Передбачення мережі: (TimeSteps, Batch, NumClasses)
            preds = model(images)
            time_steps = preds.size(0)
            batch_size = preds.size(1)

            # Всі послідовності в батчі мають однакову довжину на виході (Width)
            pred_lengths = torch.full(size=(batch_size,), fill_value=time_steps, dtype=torch.long).to(device)

            # Рахуємо CTC Loss
            loss = criterion(preds, labels, pred_lengths, target_lengths)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Епоха [{epoch + 1}/{EPOCHS}] | Помилка (CTC Loss): {epoch_loss:.4f}")

    torch.save(model.state_dict(), "crnn_plate_model.pth")
    print("\nНавчання CRNN завершено! Модель збережено як 'crnn_plate_model.pth'")


if __name__ == '__main__':
    train_model()