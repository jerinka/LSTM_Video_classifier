import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import LSTM, ConvLSTM2D
from keras.layers import Dense
from keras.layers import Flatten,Dropout
from keras.layers import TimeDistributed,GlobalAveragePooling2D


np.random.seed(0)
data = np.random.randint(low=1,high=9,size=(99, 7, 11, 11, 3))
#(samples, timeshifts, heigt, width, nchannels)
print (data.shape)
#(99, 7, 11, 11, 3)

labels = np.random.randint(low=0,high=3,size=(99)) #(samples)
print (labels.shape)
#(99,)

classes = np.unique(labels)
print (classes)
#(0,1,2)


X_train, X_test, Y_train, Y_test = train_test_split(data,labels,test_size=0.30,random_state=0)

print ('X_train', X_train.shape)
print ('X_test', X_test.shape)
print ('Y_train', Y_train.shape)
print ('Y_test', Y_test.shape)


##X_train (69, 7, 11, 11, 3)
##X_test (30, 7, 11, 11, 3)
##Y_train (69,)
##Y_test (30,)


Y_train_ = to_categorical(Y_train) 
print ('Y_train_categorical', Y_train_.shape)

##Y_train_categorical (69, 2)

input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4])
print ('input_shape', input_shape)

##input_shape (7, 11, 11, 3)
'''
model = Sequential()
model.add(ConvLSTM2D(16, kernel_size=(3,3), activation='sigmoid',padding='same',input_shape=input_shape,
                     return_sequences=True))
model.add(ConvLSTM2D(16, kernel_size=(3,3), activation='sigmoid',padding='same'))
model.add(Dense(len(classes), activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X_train, Y_train_, epochs=10, verbose=10, batch_size=32)
'''
model = Sequential()
model.add(ConvLSTM2D(16, kernel_size=(3,3), activation='sigmoid',padding='same',input_shape=input_shape,
                     return_sequences=True))
model.add(ConvLSTM2D(16, kernel_size=(3,3), activation='sigmoid',padding='same'))
model.add(Flatten())
model.add(Dense(len(classes), activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[ 'accuracy' ])

model.fit(X_train, Y_train_, epochs=20, verbose=1, batch_size=32)



