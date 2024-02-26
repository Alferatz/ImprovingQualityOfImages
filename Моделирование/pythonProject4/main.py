import cv2
import numpy as np
import os

# Создание директории для проверочных изображений
test_images_dir = 'test_images'
os.makedirs(test_images_dir, exist_ok=True)


# Функция для генерации случайного изображения с кругом или треугольником
def generate_random_shape(size):
    shape = np.zeros(size, dtype=np.uint8)  # Черный фон

    # Случайный выбор между кругом, треугольником и пустотой
    shape_type = np.random.choice(['circle', 'triangle', 'empty'])
    if shape_type == 'circle':
        center = (size[1] // 2, size[0] // 2)
        radius = min(size) // 3
        cv2.circle(shape, center, radius, 255, -1)
    elif shape_type == 'triangle':
        pts = np.array([[size[1] // 2, (size[0] // 2) - 7],
                        [(size[1] // 2) - 10, (size[0] // 2) + 7],
                        [(size[1] // 2) + 10, (size[0] // 2) + 7]], np.int32)
        cv2.fillPoly(shape, [pts], 255)
    # Если выбрана пустота, оставляем черный фон

    return shape


# Создание проверочных изображений
for i in range(10):
    big_image = np.zeros((280, 280), dtype=np.uint8)  # Черный фон для большого изображения

    # Генерация 10 маленьких случайных изображений и их объединение в одно большое изображение
    for row in range(0, 280, 28):
        for col in range(0, 280, 28):
            small_image = generate_random_shape((28, 28))
            big_image[row:row + 28, col:col + 28] = small_image

    # Сохранение проверочного изображения
    cv2.imwrite(os.path.join(test_images_dir, f'test_image_{i}.png'), big_image)

print("Генерация проверочных изображений завершена.")