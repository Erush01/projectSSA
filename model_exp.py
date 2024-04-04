import tensorflow as tf
import numpy as np
import random
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from mmt import MiniMegaTortora

mmt=MiniMegaTortora(satNumber=1)
classes=[[x] for x in mmt.satelliteData]
dataset=list()
for i in classes:
    for j in mmt.satelliteData[i[0]]:
        dataset.append(j)
        
random.shuffle(dataset)
x=list()
y=list()
for i in dataset:
    for data in i['data']:
        y.append([i['class']])
        x.append(data)
x=np.array(x,dtype=object)
# x = np.asarray(x).astype('float32')
y=np.array(y)
cat=preprocessing.OneHotEncoder().fit(classes)
y=cat.transform(y).toarray()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.25,random_state=1)
model=tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(2370,1)),
    tf.keras.layers.Conv1D(4,kernel_size=(1),activation='relu'),
    tf.keras.layers.Conv1D(4,kernel_size=(1),activation='relu'),
    tf.keras.layers.Conv1D(4,2,2,activation='relu'),
    tf.keras.layers.MaxPooling1D(2,2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1,activation='softmax')]
)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(x=x_train,
          y=y_train,
          batch_size=10,
          epochs=100,
          validation_data=(x_val,y_val),
            )

