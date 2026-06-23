import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import time

st.set_page_config(
    page_title="Система розпізнавання номерних знаків",
    page_icon="🚗",
    layout="centered"
)

CRNN_CHARS = "-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CRNN_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CRNN_NUM_CLASSES = len(CRNN_CHARS)

RESNET_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
RESNET_NUM_CLASSES = len(RESNET_CHARS)
SEQ_LENGTH = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CRNN(nn.Module):
    def __init__(self, img_channels=3, num_classes=CRNN_NUM_CLASSES):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, kernel_size=2, padding=0), nn.ReLU(),
        )
        self.rnn = nn.Sequential(
            nn.LSTM(1536, 256, bidirectional=True, batch_first=True),
        )
        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        batch, channels, height, width = conv.size()
        conv = conv.permute(0, 3, 1, 2).contiguous()
        conv = conv.view(batch, width, channels * height)
        rnn_out, _ = self.rnn(conv)
        output = self.fc(rnn_out)
        output = output.permute(1, 0, 2)
        return torch.nn.functional.log_softmax(output, dim=2)


class MultiHeadResNet(nn.Module):
    def __init__(self):
        super(MultiHeadResNet, self).__init__()
        resnet = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc_heads = nn.ModuleList([
            nn.Linear(512, RESNET_NUM_CLASSES) for _ in range(SEQ_LENGTH)
        ])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        outputs = [head(x) for head in self.fc_heads]
        return outputs


@st.cache_resource
def load_models():
    crnn_model = CRNN().to(device)
    try:
        crnn_model.load_state_dict(torch.load("crnn_plate_model.pth", map_location=device))
        crnn_model.eval()
        crnn_status = "Успішно"
    except Exception as e:
        crnn_status = f"Помилка: {e}"

    resnet_model = MultiHeadResNet().to(device)
    try:
        resnet_model.load_state_dict(torch.load("resnet_plate_model.pth", map_location=device))
        resnet_model.eval()
        resnet_status = "Успішно"
    except Exception as e:
        resnet_status = f"Помилка: {e}"

    return crnn_model, resnet_model, crnn_status, resnet_status


def decode_ctc(pred_indices):
    result = []
    prev_idx = -1
    for idx in pred_indices:
        if idx != 0 and idx != prev_idx:
            result.append(CRNN_ALPHABET[idx - 1])
        prev_idx = idx
    return "".join(result)


def main():
    st.title("🚗 Система розпізнавання автомобільних номерів")
    st.markdown("Завантажте попередньо локалізоване зображення номерного знака для розпізнавання.")

    crnn_model, resnet_model, crnn_status, resnet_status = load_models()

    st.sidebar.header("Налаштування системи")
    model_choice = st.sidebar.selectbox(
        "Оберіть нейромережеву архітектуру:",
        ("Гібридна модель (CRNN + CTC)", "Базова згорткова мережа (CNN)")
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Статус моделей:")
    st.sidebar.write(f"**CRNN:** {crnn_status}")
    st.sidebar.write(f"**ResNet18:** {resnet_status}")

    uploaded_file = st.file_uploader("Оберіть зображення (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Вхідне зображення", use_container_width=True)

        if st.button("Розпізнати текст", type="primary", use_container_width=True):
            start_time = time.time()

            try:
                with torch.no_grad():
                    if model_choice == "Гібридна модель (CRNN + CTC)":
                        transform_crnn = transforms.Compose([
                            transforms.Resize((64, 256)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5], std=[0.5])
                        ])
                        img_tensor = transform_crnn(image).unsqueeze(0).to(device)

                        preds = crnn_model(img_tensor)
                        pred_indices = preds.argmax(dim=2).squeeze(1).tolist()
                        prediction = decode_ctc(pred_indices)

                    else:
                        transform_resnet = transforms.Compose([
                            transforms.Resize((64, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                        img_tensor = transform_resnet(image).unsqueeze(0).to(device)

                        outputs = resnet_model(img_tensor)
                        predicted_chars = []
                        for out in outputs:
                            _, predicted = torch.max(out.data, 1)
                            char_idx = predicted.item()
                            predicted_chars.append(RESNET_CHARS[char_idx])
                        prediction = "".join(predicted_chars)

                inference_time = (time.time() - start_time) * 1000

                st.success("Розпізнавання завершено успішно!")
                st.markdown(f"### Результат: **{prediction}**")
                st.info(f"Використана архітектура: {model_choice}\n\nЧас обчислення: {inference_time:.1f} мс")

            except Exception as e:
                st.error(f"Помилка обробки: {e}")


if __name__ == "__main__":
    main()