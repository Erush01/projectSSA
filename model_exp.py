
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Silence tensorflow
import tensorflow as tf
import numpy as np
import random
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import datetime
from mmt import MiniMegaTortora

#Tensorflow settings

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


mmt=MiniMegaTortora(satNumber=20)
classes=[[x] for x in mmt.satelliteData]#Main classes
dataset=list()
#Add all satellites from each class to bigger set
for i in classes:
    for j in mmt.satelliteData[i[0]]:
        dataset.append(j)

#Shuffle dataset
np.random.shuffle(dataset)
x=list()
y=list()
trackSize=1000 #Maximum track size

#Parse dataset into tracks and classes
for i in dataset:
    for data in i['data']:
        y.append([i['class']])
        x.append(data[:700])
        
#Pad tracks to maximum TrackSize
x=[np.pad(x,((0,700-len(x))),mode='constant') for x in x]

#Numpy array conversion        
x=np.array([np.array(val) for val in x])
y=np.array(y)



#One hot encoding for labels
cat=preprocessing.OneHotEncoder().fit(classes)
y=cat.transform(y).toarray()

#Train-Val-Test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,
                                               shuffle=True,
                                               random_state=random.randint(0,100))
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,
                                             shuffle=True,
                                             test_size=0.2,
                                             random_state=random.randint(0,100))

# Normalization
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_val=scaler.fit_transform(x_val)
x_test=scaler.fit_transform(x_test)


print(x_train.shape)


#Expanding dimension to fit Convolutional layer
x_train=tf.expand_dims(x_train,axis=-1)
x_val=tf.expand_dims(x_val,axis=-1)
x_test=tf.expand_dims(x_test,axis=-1)

#Model 
model=tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(x_train.shape[1],x_train.shape[2])),
    tf.keras.layers.Conv1D(32,kernel_size=2,activation='relu'),
    tf.keras.layers.MaxPool1D(pool_size=(2)),
    tf.keras.layers.Conv1D(32,kernel_size=2,activation='relu'),
    tf.keras.layers.MaxPool1D(pool_size=(2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64,kernel_size=2,activation='relu'),
    tf.keras.layers.MaxPool1D(pool_size=(2)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(3,activation='softmax')]
)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['acc'])

model.summary()

#Logging metrics using tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


#Training model
history=model.fit(x=x_train,
          y=y_train,
          batch_size=10,
          epochs=50,
          validation_data=(x_val,y_val),
          callbacks=[tensorboard_callback]
            )

