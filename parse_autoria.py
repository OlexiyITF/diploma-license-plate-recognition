import os
import shutil

#налаштування шляхів
RIA_IMAGES_DIR = r'E:\Program_works\trash_zone_for_diploma\train\img'
DEST_DIR = r'E:\Program_works\Diploma_work\dataset\ukr_plates\train\images'

#Словник
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def main():
    print("Скануємо назви файлів у папці img...")

    custom_plates_copied = 0

    for filename in os.listdir(RIA_IMAGES_DIR):
        if not filename.endswith('.png') and not filename.endswith('.jpg'):
            continue

        # Формат файлу: 13873090--17-PILOT--Mitsubishi...png
        parts = filename.split('--')
        if len(parts) < 3:
            continue

        region_and_plate = parts[1]  # Наприклад: "17-PILOT"

        if '-' not in region_and_plate:
            continue

        # Відрізаємо код регіону, залишаємо текст номера
        raw_text = region_and_plate.split('-', 1)[1]  # "PILOT"

        # Очищаємо текст від пробілів та спецсимволів
        clean_text = str(raw_text).upper().replace(" ", "").replace("_", "")

        # критичний фільтир: залишаємо тільки ті номери, які є в нашому словнику
        is_valid_chars = all(c in CHARS for c in clean_text)

        if not is_valid_chars or len(clean_text) == 0:
            continue

        # Шукаємо іменні (не 8 символів) або нові формати DI/PD
        is_custom = len(clean_text) != 8
        is_new_format = clean_text.startswith("DI") or clean_text.startswith("PD")

        if is_custom or is_new_format:
            src_path = os.path.join(RIA_IMAGES_DIR, filename)

            # Створюємо правильну назву для нашого датасету: PILOT.png
            ext = os.path.splitext(filename)[1]
            new_filename = f"{clean_text}{ext}"
            dst_path = os.path.join(DEST_DIR, new_filename)

            if not os.path.exists(dst_path):
                shutil.copy(src_path, dst_path)
                custom_plates_copied += 1

            if custom_plates_copied >= 500:
                break

    print(f"\nГотово! У твій датасет успішно додано чистих латинських кастомних номерів: {custom_plates_copied}")
    print("Тепер сміливо запускай train_crnn.py!")


if __name__ == '__main__':
    main()