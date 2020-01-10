import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, LSTM, Lambda, ConvLSTM2D,  GRU  # ,CuDNNGRU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip header
    for line in reader:
        lines.append(line)
images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = '../data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)
X_train = np.array(images)
y_train = np.array(measurements)
batch_size = 10
look_back = 3
model = Sequential()
# model.add(GRU(1, return_sequences=False, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255.) - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(ConvLSTM2D(30, 5, 5,  return_sequences=False))
# model.add(Convolutional2D(48, 5, 5,subsambple=(2,2) activation='relu'))
# model.add(Convolutional2D(64, 3, 3,subsambple=(2,2) activation='relu'))
# model.add(Convolutional2D(64, 3, 3,subsambple=(2,2) activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(GRU(30, batch_input_shape=(look_back, 1),
              stateful=True, return_sequences=False))
model.add(Flatten())
#model.add(GRU(1, return_sequences=False))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit_generator(X_train, y_train, validation_split=0.32,
                    shuffle=True, epochs=7)
model.save('model.h5')
