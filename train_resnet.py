import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np


TRAIN_DIR = r'dataset\ukr_plates\train\images'
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS)
SEQ_LENGTH = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Використовується пристрій: {device}")


transform = T.Compose([
    T.Resize((64, 224)),
    T.RandomRotation(degrees=5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class LicensePlateDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = []

        for f in os.listdir(img_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                label = os.path.splitext(f)[0].upper()
                label = ''.join(c for c in label if c in CHARS)
                if len(label) == SEQ_LENGTH:
                    self.img_names.append(f)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label_str = os.path.splitext(img_name)[0].upper()
        label_str = ''.join(c for c in label_str if c in CHARS)
        label_idx = [CHAR2IDX[c] for c in label_str]

        return image, torch.tensor(label_idx, dtype=torch.long)


class MultiHeadResNet(nn.Module):
    def __init__(self):
        super(MultiHeadResNet, self).__init__()

        resnet = models.resnet18(weights=None)

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        self.fc_heads = nn.ModuleList([
            nn.Linear(512, NUM_CLASSES) for _ in range(SEQ_LENGTH)
        ])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        outputs = [head(x) for head in self.fc_heads]
        return outputs


def train_model():
    print("Завантаження датасету з аугментацією...")
    train_dataset = LicensePlateDataset(TRAIN_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Знайдено фотографій: {len(train_dataset)}")

    model = MultiHeadResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nПочинаємо навчання ResNet18...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_chars = 0
        total_chars = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = 0
            for i in range(SEQ_LENGTH):
                loss += criterion(outputs[i], labels[:, i])

                _, predicted = torch.max(outputs[i].data, 1)
                total_chars += labels.size(0)
                correct_chars += (predicted == labels[:, i]).sum().item()

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = (correct_chars / total_chars) * 100
        print(f"Епоха [{epoch + 1}/{EPOCHS}] | Помилка: {epoch_loss:.4f} | Точність символів: {epoch_acc:.2f}%")

    torch.save(model.state_dict(), "resnet_plate_model.pth")
    print("\nНавчання завершено! Модель збережено як 'resnet_plate_model.pth'")


if __name__ == '__main__':
    train_model()