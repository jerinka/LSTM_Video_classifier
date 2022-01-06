import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, utils, applications

#Assuming there are 5 audio files
num_audio = 5
data = []

#Create a random number of mel-spectrograms for each audio file
for i in range(num_audio):
    n_images = np.random.randint(4,10)
    data.append(np.random.random((n_images,128,216,1)))
    
print([i.shape for i in data])
#import pdb;pdb.set_trace()
#Convert each set of images (for each audio) to tensors and then a ragged tensor
tensors = [tf.convert_to_tensor(i) for i in data]
X_train = tf.ragged.stack(tensors).to_tensor()

#Creating dummy y_train, one for each audio files
y_train = tf.convert_to_tensor(np.random.randint(0,2,(5,2)))


#Create model
inp = layers.Input((None,128,216,1), ragged=True)

cnn = tf.keras.applications.DenseNet169(include_top=True, 
                                                weights=None, 
                                                input_tensor=None, 
                                                input_shape=(128,216,1), #<----- input shape for cnn is just the image
                                                pooling=None, classes=2)


#Feel free to modify these layers!
x = layers.TimeDistributed(cnn)(inp)
x = layers.LSTM(8)(x)
out = layers.Dense(2)(x)

model = Model(inp, out)
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics='accuracy')

utils.plot_model(model, show_shapes=True, show_layer_names=False)



model.fit(X_train, y_train, epochs=2)




