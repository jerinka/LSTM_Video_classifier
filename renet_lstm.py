from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten,Dropout,Input
from keras.layers import TimeDistributed,GlobalAveragePooling2D
from keras.models import Model

from keras.applications.vgg16 import VGG16,ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np



#model = VGG16(weights='imagenet', include_top=False)


resnet = ResNet50(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
'''
input_layer = Input(shape=(seq_len, 224, 224, 3))
curr_layer = TimeDistributed(resnet)(input_layer)
curr_layer = Reshape(target_shape=(seq_len, 2048))(curr_layer)
lstm_out = LSTM(128)(curr_layer)
'''

# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(200, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')








