import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.datasets import mnist
import os
import cv2
import numpy as np

from keras.models import load_model


# Путь к папкам с обучающими и тестовыми изображениями
train_data_dir = 'C:/pythonProject3/train_data'
test_data_dir = 'C:/pythonProject3/test_data'
# train_data_dir = 'C:/pythonProject3/train_data_min'
# test_data_dir = 'C:/pythonProject3/test_data_min'
test_data_dir_280x280 = 'C:/pythonProject4/test_images'
# test_data_dir_280x280 = 'C:/pythonProject5/test_images'


# Функция для загрузки изображений из папки
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)  # Чтение изображения в оттенках серого
        if img is not None:
            images.append(img)
    return images

# Загрузка обучающих и тестовых изображений
x_train_images = load_images_from_folder(train_data_dir)
x_test_images = load_images_from_folder(test_data_dir)


# Преобразование изображений в массивы numpy
x_train = np.array(x_train_images)
x_test = np.array(x_test_images)

# Печать размерности массивов
print("Размерность x_train:", x_train.shape)
print("Размерность x_test:", x_test.shape)

x = 28
y = 28

# (x_train, _), (x_test, _) = mnist.load_data()

# normalize the image data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# # Пример использования
print("Размер обучающего набора данных:", x_train.shape)
print("Размер тестового набора данных:", x_test.shape)
# reshape in the input data for the model
x_train = x_train.reshape(len(x_train), x, y, 1)
x_test = x_test.reshape(len(x_test), x, y, 1)
x_test.shape
# add noise
noise_factor = 0.6
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
# clip the values in the range of 0-1
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
# randomly select input image
index = np.random.randint(len(x_test))
# plot the image
plt.imshow(x_test[index].reshape(x,y))
plt.gray()
plt.show()
# randomly select input image
index = np.random.randint(len(x_test))
# plot the image
plt.imshow(x_test_noisy[index].reshape(x,y))
plt.gray()
plt.show()
# randomly select input image
index = np.random.randint(len(x_test))
# plot the image
plt.imshow(x_test_noisy[index].reshape(x,y))
plt.gray()
plt.show()


# model = Sequential([
#                     # encoder network
#                     Conv2D(32, 3, activation='relu', padding='same', input_shape=(x, y, 1)),
#                     MaxPooling2D(2, padding='same'),
#                     Conv2D(16, 3, activation='relu', padding='same'),
#                     MaxPooling2D(2, padding='same'),
#                     # decoder network
#                     Conv2D(16, 3, activation='relu', padding='same'),
#                     UpSampling2D(2),
#                     Conv2D(32, 3, activation='relu', padding='same'),
#                     UpSampling2D(2),
#                     # output layer
#                     Conv2D(1, 3, activation='sigmoid', padding='same')
# ])
#
# model.compile(optimizer='adam', loss='binary_crossentropy')
# model.summary()
# # train the model
# model.fit(x_train_noisy, x_train, epochs=10, batch_size=256, validation_data=(x_test_noisy, x_test))

# Сохранение модели
# model.save('my_model_0601_2_minRazm.h5')

# # Загрузка модели
model = load_model('my_model_0601.h5')
# model = load_model('my_model_0601_2_minRazm.h5')
'''Вариант отображения 1'''
# randomly select input image
index = np.random.randint(len(x_test))
# plot the image
plt.imshow(x_test_noisy[index].reshape(x,y))
plt.gray()
plt.show()
# predict the results from model (get compressed images)
pred = model.predict(x_test_noisy)
# visualize compressed image
plt.imshow(pred[index].reshape(x,y))
plt.gray()
plt.show()
index = np.random.randint(len(x_test))
plt.figure(figsize=(10, 4))
# display original image
ax = plt.subplot(1, 2, 1)
plt.imshow(x_test_noisy[index].reshape(x,y))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
# display compressed image
ax = plt.subplot(1, 2, 2)
plt.imshow(pred[index].reshape(x,y))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()

'''Вариант отображения 2'''
# Загрузка исходного изображения размером 280x280
# original_image = load_images_from_folder(test_data_dir_280x280)
# x_test_280x280 = np.array(original_image)
# x_test_280x280 = x_test_280x280.astype('float32') / 255
# # index = np.random.randint(len(x_test_280x280))
# # plot the image
# plt.imshow(x_test_280x280[1].reshape(280,280))
# plt.gray()
# plt.show()
#
# # Зашумление исходного изображения (эмуляция)
# noise_factor = 0.6  # Параметр шума (можете изменить по вашему усмотрению)
# x_test_280x280_noisy = x_test_280x280 + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_280x280.shape)
# plt.imshow(x_test_280x280_noisy[1].reshape(280,280))
# plt.gray()
# plt.show()
#
# # Разделение массива на 100 кусочков размером 28x28
# pieces = []
#
# print(x_test_280x280_noisy.shape)
#
# for i in range(10):
#     for j in range(10):
#         piece = x_test_280x280_noisy[1, i*28:(i+1)*28, j*28:(j+1)*28]
#         pieces.append(piece)
#
# # Проверка размерности каждого кусочка
# for piece in pieces:
#     print(piece.shape)
#
# # Предсказание и убирание шумов для каждого кусочка
# denoised_pieces = []
# for piece in pieces:
#     # Пропуск кусочка через модель для убирания шумов
#     denoised_piece = model.predict(piece.reshape(1, 28, 28, 1))
#     denoised_pieces.append(denoised_piece)
#
# plt.imshow(denoised_pieces[1].reshape(x,y))
# plt.gray()
# plt.show()
# # Инициализируем счетчики для строк и столбцов в новом массиве
# row_counter = 0
# col_counter = 0
# array_280x280 = np.zeros((280, 280))
#
# # Обходим все квадратики и помещаем их в соответствующие места в новом массиве
# for piece in denoised_pieces:
#     piece = np.squeeze(piece)  # Удаляем размерность 1
#     array_280x280[row_counter:row_counter+28, col_counter:col_counter+28] = piece
#     col_counter += 28  # Переходим к следующему столбцу
#     if col_counter == 280:  # Если достигли конца строки, переходим на новую строку
#         col_counter = 0
#         row_counter += 28
#
# # Отображение исходного изображения, зашумленного изображения и восстановленного изображения
# plt.figure(figsize=(12, 6))
#
# plt.subplot(1, 3, 1)
# plt.imshow(x_test_280x280[1].reshape(280,280), cmap='gray')
# plt.title('Original Image')
#
# plt.subplot(1, 3, 2)
# plt.imshow(x_test_280x280_noisy[1].reshape(280,280), cmap='gray')
# plt.title('Noisy Image')
#
# plt.subplot(1, 3, 3)
# plt.imshow(array_280x280, cmap='gray')
# plt.title('Denoised Image')
#
# plt.show()

