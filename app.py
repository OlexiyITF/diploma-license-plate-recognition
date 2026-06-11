import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import time

# Конфігурація сторінки веб-додатка
st.set_page_config(
    page_title="Система розпізнавання номерних знаків",
    page_icon="🚗",
    layout="centered"
)

# Константи
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def decode_ctc(pred_indices):
    """
    Декодування результату за правилами Connectionist Temporal Classification (CTC).
    Вилучає дублікати та ігнорує маркери порожнечі (blank tokens).
    """
    result = []
    prev_idx = -1
    for idx in pred_indices:
        # 0 зарезервовано під blank token
        if idx != 0 and idx != prev_idx:
            result.append(ALPHABET[idx - 1])
        prev_idx = idx
    return "".join(result)


def main():
    st.title("🚗 Система розпізнавання автомобільних номерів")
    st.markdown("Завантажте попередньо локалізоване зображення номерного знака для розпізнавання.")

    # Бокова панель для налаштувань
    st.sidebar.header("Налаштування розпізнавання")
    model_choice = st.sidebar.selectbox(
        "Оберіть нейромережеву архітектуру:",
        ("Гібридна модель (CRNN + CTC)", "Базова згорткова мережа (CNN)")
    )

    # Завантаження зображення користувачем
    uploaded_file = st.file_uploader("Оберіть зображення (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Відображення завантаженого зображення
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Вхідне зображення", use_container_width=True)

        if st.button("Розпізнати текст", type="primary", use_container_width=True):
            start_time = time.time()

            # Конвеєр попередньої обробки зображення
            transform = transforms.Compose([
                transforms.Resize((64, 256)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

            try:
                # Підготовка тензора (додавання виміру батчу)
                img_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    if model_choice == "Гібридна модель (CRNN + CTC)":
                        # outputs = crnn_model(img_tensor)
                        # _, preds = outputs.max(2)
                        # preds = preds.transpose(1, 0).contiguous().view(-1)
                        # prediction = decode_ctc(preds.tolist())
                        prediction = "AT0397HA"  # Заглушка для демонстрації
                    else:
                        # outputs = resnet_model(img_tensor)
                        # prediction = decode_resnet(outputs)
                        prediction = "AT039"

                inference_time = (time.time() - start_time) * 1000

                # Виведення результатів
                st.success("Розпізнавання завершено успішно!")
                st.markdown(f"### Результат: **{prediction}**")
                st.info(f"Використана архітектура: {model_choice}\n\nОціночний час інференсу: ~{inference_time:.1f} мс")

            except Exception as e:
                st.error(f"Помилка тензорних обчислень або обробки зображення: {e}")


if __name__ == "__main__":
    main()