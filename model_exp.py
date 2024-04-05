import tensorflow as tf
import numpy as np
import random
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import datetime

from mmt import MiniMegaTortora
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

mmt=MiniMegaTortora(satNumber=6)
classes=[[x] for x in mmt.satelliteData]
dataset=list()
for i in classes:
    for j in mmt.satelliteData[i[0]]:
        dataset.append(j)
        
random.shuffle(dataset)
x=list()
y=list()
trackSize=700
for i in dataset:
    for data in i['data']:
        y.append([i['class']])
        x.append(data[:700])

x=[np.pad(x,((0,700-len(x))),mode='constant') for x in x]        
x=np.array([np.array(val) for val in x])
y=np.array(y)


cat=preprocessing.OneHotEncoder().fit(classes)
y=cat.transform(y).toarray()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,shuffle=True,random_state=42)
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,shuffle=True,test_size=0.2,random_state=31)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_val=scaler.fit_transform(x_val)
x_test=scaler.fit_transform(x_test)

print(y_train[:10])
x_train=tf.expand_dims(x_train,axis=-1)
x_val=tf.expand_dims(x_val,axis=-1)
x_test=tf.expand_dims(x_test,axis=-1)

model=tf.keras.models.Sequential([
    # tf.keras.layers.InputLayer(input_shape=),
    tf.keras.layers.Conv1D(32,kernel_size=2,activation='relu',input_shape=(x_train.shape[1],x_train.shape[2])),
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
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


history=model.fit(x=x_train,
          y=y_train,
          batch_size=10,
          epochs=50,
          validation_data=(x_val,y_val),
          callbacks=[tensorboard_callback]
            )

