from random import random
from random import randint
from numpy import array
from numpy import zeros
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten,Dropout
from keras.layers import TimeDistributed,GlobalAveragePooling2D

timesteps = 5
width  = 100
height = 100

def get_clstm():
    model = Sequential()
    model.add(TimeDistributed(Conv2D(2, (2,2), activation= 'relu' ), input_shape=(None,width,height,1)))
    x = ConvLSTM2D(nb_filter=c,nb_row=3,nb_col=3,border_mode='same',return_sequences=True)(input_img)

def get_lstm_model1():
    model = Sequential()
    model.add(TimeDistributed(Conv2D(2, (2,2), activation= 'relu' ), input_shape=(None,width,height,1)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50))
    model.add(Dense(1, activation= 'sigmoid' ))
    model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
    print(model.summary())
    return model

def get_lstm_model2():
    model = Sequential()# input, with 64 convolutions for 5 images
    model.add(TimeDistributed(Conv2D(64, (3,3), padding='same', strides=(2,2), activation='relu'),
    input_shape = (5, 100, 100, 1) ))
    model.add(TimeDistributed(MaxPooling2D((2,2), strides=(2,2)) ))
    model.add(TimeDistributed( Conv2D(64, (3,3), padding='same', strides=(2,2), activation='relu') ))
    model.add(TimeDistributed(MaxPooling2D((2,2), strides=(2,2)) ))
    model.add(TimeDistributed(Conv2D(128, (3,3),padding='same', strides=(2,2), activation='relu') ))
    model.add(TimeDistributed(MaxPooling2D((2,2), strides=(2,2)) ))
    model.add(TimeDistributed( Conv2D(128, (3,3),padding='same', strides=(2,2), activation='relu')))
    model.add(TimeDistributed(GlobalAveragePooling2D()  ))
    model.add(LSTM(100, activation='relu', return_sequences=False))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(.5))# For example, for 3 outputs classes 
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
    print(model.summary())
    return model

# generate the next frame in the sequence
def next_frame(last_step, last_frame, column):
    # define the scope of the next step
    lower = max(0, last_step-1)
    upper = min(last_frame.shape[0]-1, last_step+1)
    # choose the row index for the next step
    step = randint(lower, upper)
    # copy the prior frame
    frame = last_frame.copy()
    # add the new step
    frame[step, column] = 1
    return frame, step
    
# generate a sequence of frames of a dot moving across an image
def build_frames(timesteps,width,height, ch=1):
    frames = list()
    # create the first frame
    frame = zeros((width,height))
    step = randint(0, timesteps-1)
    # decide if we are heading left or right
    right = 1 if random() < 0.5 else 0
    col = 0 if right else timesteps-1
    frame[step, col] = 1
    frames.append(frame)
    # create all remaining frames
    for i in range(1, timesteps):
        col = i if right else timesteps-1-i
        frame, step = next_frame(step, frame, col)
        frames.append(frame)
    return frames, right
        
# generate multiple sequences of frames and reshape for network input
def generate_examples(timesteps,width,height, n_patterns):
    X, y = list(), list()
    for _ in range(n_patterns):
        frames, right = build_frames(timesteps,width,height, ch=1)
        X.append(frames)
        y.append(right)
    # resize as [samples, timesteps, width, height, channels]
    X = array(X).reshape(n_patterns, timesteps,width,height, 1)
    y = array(y).reshape(n_patterns, 1)
    return X, y
    
model = get_lstm_model2()

# fit model
X, y = generate_examples(timesteps,width,height, 50)
#import pdb;pdb.set_trace()
model.fit(X, y, batch_size=1, epochs=10)

# evaluate model
X, y = generate_examples(timesteps,width,height, 10)
loss, acc = model.evaluate(X, y, verbose=0)
print( 'loss:',loss, 'acc:', acc*100)

# prediction on new data
X, y = generate_examples(timesteps,width,height, 1)
yhat = model.predict_classes(X, verbose=0)
expected = "Right" if y[0]==1 else "Left"
predicted = "Right" if yhat[0]==1 else "Left"
print( 'Expected: %s, Predicted: %s', (expected, predicted))


