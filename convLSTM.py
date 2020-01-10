import numpy as np, scipy.ndimage, matplotlib.pyplot as plt
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten,Input,TimeDistributed,BatchNormalization
from keras.layers import Convolution2D, ConvLSTM2D, MaxPooling2D, UpSampling2D,GlobalAveragePooling2D,AveragePooling3D,Reshape
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
np.random.seed(123)

raw = np.arange(96).reshape(8,3,4)
data1 = scipy.ndimage.zoom(raw, zoom=(1,100,100), order=1, mode='nearest') #low res
print (data1.shape)
#(8, 300, 400)

data2 = np.arange(9)
print (data2.shape)
#(8, 300, 400)

X_train = data1.reshape(1, data1.shape[0], data1.shape[1], data1.shape[2], 1)
Y_train = data2.reshape(1, data2.shape[0])
#(samples,time, rows, cols, channels)
print (X_train.shape)
print (Y_train.shape)
import pdb;pdb.set_trace()

def getmodel1():
    model = Sequential()
    input_shape = (data1.shape[0], data1.shape[1], data1.shape[2], 1)
    #samples, time, rows, cols, channels
    model.add(ConvLSTM2D(16, kernel_size=(3,3), activation='sigmoid',padding='same',input_shape=input_shape,
                         return_sequences=True))
    model.add(ConvLSTM2D(8, kernel_size=(3,3), activation='sigmoid',padding='same'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(10, activation='softmax'))  # output shape: (None, 10)
    print (model.summary())
    return model


'''
x = Input(shape=(300, 400, 8))
y = GlobalAveragePooling2D()(x)
y = Dense(10, activation='softmax')(y)
classifier = Model(inputs=x, outputs=y)

x = Input(shape=(data1.shape[0], data1.shape[1], data1.shape[2], 1))
y = ConvLSTM2D(16, kernel_size=(3, 3),
               activation='sigmoid',
               padding='same',
               return_sequences=True)(x)
y = ConvLSTM2D(8, kernel_size=(3, 3),
               activation='sigmoid',
               padding='same',
               return_sequences=True)(y)
y = TimeDistributed(classifier)(y)  # output shape: (None, 8, 10)

model = Model(inputs=x, outputs=y)

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
'''

def getmodel2():
    model = Sequential()
    input_shape = (data1.shape[0], data1.shape[1], data1.shape[2], 1)
    model.add(ConvLSTM2D(16, kernel_size=(3, 3), activation='sigmoid', padding='same',
                         input_shape=input_shape,
                         return_sequences=True))
    model.add(ConvLSTM2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same',
                         return_sequences=True))

    model.compile(loss='mse', optimizer='adam')
    return model
   
def getmodel3(): 
    model = Sequential()

    model.add(ConvLSTM2D(
            filters=40,
            kernel_size=(3, 3),
            input_shape=(None, 300, 400, 1),
            padding='same',
            return_sequences=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(
            filters=40,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(
            filters=40,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=True))
    model.add(BatchNormalization())

    model.add(AveragePooling3D((1, 135, 240)))
    model.add(Reshape((-1, 40)))
    model.add(Dense(
            units=9,
            activation='sigmoid'))

    model.compile(
            loss='categorical_crossentropy',
            optimizer='adadelta'
    )
    print(model.summary())
    return model


model = getmodel3()
model.fit(X_train, Y_train, 
          batch_size=1, epochs=10, verbose=1)

model.save('model.h5')
import pdb;pdb.set_trace()

x,y = model.evaluate(X_train, Y_train, verbose=0)
print (x,y)



