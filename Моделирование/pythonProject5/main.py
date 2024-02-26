import cv2
import numpy as np
import os

# Создание директории для проверочных изображений
test_images_dir = 'test_images'
os.makedirs(test_images_dir, exist_ok=True)


# Функция для генерации маленького случайного изображения с квадратом, треугольником и кругом
def generate_random_shape_small(size):
    shape = np.zeros(size, dtype=np.uint8)  # Черный фон

    # Случайный выбор между квадратом, треугольником, кругом и пустотой
    shape_type = np.random.choice(['square', 'triangle', 'circle', 'empty'])
    if shape_type == 'square':
        side_length = min(size) // 6
        cv2.rectangle(shape, (size[1] // 6 - side_length // 6, size[0] // 6 - side_length // 6),
                      (size[1] // 6 + side_length // 6, size[0] // 6 + side_length // 2), 255, -1)
    elif shape_type == 'triangle':
        pts = np.array([[size[1] // 6, (size[0] // 6) - 7],
                        [(size[1] // 6) - 10, (size[0] // 6) + 7],
                        [(size[1] // 6) + 10, (size[0] // 6) + 7]], np.int32)
        cv2.fillPoly(shape, [pts], 255)
    elif shape_type == 'circle':
        center = (size[1] // 6, size[0] // 4)
        radius = min(size) // 6
        cv2.circle(shape, center, radius, 255, -1)
    # Если выбрана пустота, оставляем черный фон


    return shape


# Создание проверочных маленьких изображений
for i in range(10):
    big_image = np.zeros((280, 280), dtype=np.uint8)  # Черный фон для большого изображения

    # Генерация 100 маленьких случайных изображений и их объединение в одно большое изображение
    for row in range(0, 280, 28):
        for col in range(0, 280, 28):
            small_image = generate_random_shape_small((28, 28))
            big_image[row:row + 28, col:col + 28] = small_image

    # Сохранение проверочного маленького изображения
    cv2.imwrite(os.path.join(test_images_dir, f'test_image_small_{i}.png'), big_image)

print("Генерация проверочных маленьких изображений завершена.")
