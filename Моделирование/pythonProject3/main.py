import cv2
import numpy as np
import os

# Создание директорий для обучающих и тестовых данных
train_dir = 'train_data'
test_dir = 'test_data'
# train_dir = 'train_data_min'
# test_dir = 'test_data_min'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Параметры изображения
image_size = (28, 28)  # Размер изображения 28x28 пикселей
num_train_images = 8000  # Количество обучающих изображений
num_test_images = 2000  # Количество тестовых изображений


# Функция для генерации изображения с фигурой
def generate_image(shape_type):
    image = np.zeros(image_size, dtype=np.uint8)  # Черный фон

    if shape_type == 'circle':
        center = (image_size[1] // 2, image_size[0] // 2)
        radius = min(image_size) // 3
        cv2.circle(image, center, radius, 255, -1)
    elif shape_type == 'square':
        side_length = min(image_size) // 2
        top_left = ((image_size[1] - side_length) // 2, (image_size[0] - side_length) // 2)
        bottom_right = (top_left[0] + side_length, top_left[1] + side_length)
        cv2.rectangle(image, top_left, bottom_right, 255, -1)
    elif shape_type == 'triangle':
        pts = np.array([[image_size[1] // 2, (image_size[0] // 2) - 7],
                        [(image_size[1] // 2) - 10, (image_size[0] // 2) + 7],
                        [(image_size[1] // 2) + 10, (image_size[0] // 2) + 7]], np.int32)
        cv2.fillPoly(image, [pts], 255)

    return image


# Генерация и сохранение обучающих изображений
for i in range(num_train_images):
    shape_type = np.random.choice(['circle', 'square', 'triangle'])
    image = generate_image(shape_type)
    cv2.imwrite(os.path.join(train_dir, f'{i}.png'), image)

# Генерация и сохранение тестовых изображений
for i in range(num_test_images):
    shape_type = np.random.choice(['circle', 'square', 'triangle'])
    image = generate_image(shape_type)
    cv2.imwrite(os.path.join(test_dir, f'{i}.png'), image)

print("Генерация изображений завершена.")
