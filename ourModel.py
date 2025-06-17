import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Silence tensorflow
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CUDNN_RESET_RND_GEN_STATE']='1'
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report
from mmt import MiniMegaTortoraDataset
import seaborn as sns
import pandas as pd
from rich import print as print_rich
from io import BytesIO
from ssaUtils import get_summary_str,train_table,DiscreteWaveletTransform,DiscreteWaveletTransform1,DelayedExponentialDecay,save_evaluated_lc_plots,pad_to_size_interpolate,pad_to_size_constant
import  argparse
#Tensorflow settings
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

physical_devices = tf.test.gpu_device_name()
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"


# Set the path where you want to save the best weights
checkpoint_path = "checkpoints/ourModel.keras"

# Create a ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_acc',       # Monitor validation accuracy
    save_best_only=True,          # Save only when val_accuracy improves
    save_weights_only=False,       # Save only the weights, not the entire model
    mode='max',                   # Maximize val_accuracy
    verbose=1
)


parser=argparse.ArgumentParser(
    prog="SSA Classifier",
)
parser.add_argument('-d','--debug',action="store_true")
args=parser.parse_args()

if(args.debug): 
    satelliteNumber=[1,1,5]
else:
    satelliteNumber=[60,160,300]



#-------------Configurations---------------------------
# satelliteNumber=[1,1,1]
trackSize = 700      # Maximum sample points for each track
epochs = 100    # Number of epochs for training
batchSize = 32        # batch size for training
buffer = BytesIO()
buffer2=BytesIO()
acc_buffer=BytesIO()
loss_buffer=BytesIO()


#--------------------Learning Rate Scheduler-------------------------------

learning_rate=1e-4
final_learning_rate = 5e-6
start_epoch=60
learning_rate_decay_factor = (final_learning_rate /learning_rate)**(1/(epochs-start_epoch))

#-------------------------------------------------------

mmt=MiniMegaTortoraDataset(satNumber=satelliteNumber,periodic=True)

x,y=mmt.load_data_new()
classes=[[x] for x in np.unique(y)]


x=DiscreteWaveletTransform1(x)
#x = DiscreteWaveletTransform(x, wavelet='db4',level=1)
#x=[pad_to_size_interpolate(array,trackSize) for array in x]
#x=[np.pad(x,((0,trackSize-len(x))),mode='constant') for x in x]
x = [pad_to_size_constant(arr, trackSize) for arr in x]

#Numpy array conversion        
x=np.array(x)
#x=np.array([np.array(val) for val in x])
y=np.array(y)
y=np.expand_dims(y,axis=-1)

cat=preprocessing.OneHotEncoder().fit(classes)
y=cat.transform(y).toarray()


# Train-Val-Test split
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               shuffle=True,
                                               test_size=0.2,
                                               stratify=y)


x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,
                                             shuffle=True,
                                             test_size=0.2,
                                             stratify=y_train)


# Normalization
scaler=StandardScaler()

x_train=scaler.fit_transform(x_train)
x_val=scaler.transform(x_val)
x_test=scaler.transform(x_test)

x_train=np.array(x_train)
x_val=np.array(x_val)
x_test=np.array(x_test)


#Only use if you not use ConvLSTM
x_train=np.expand_dims(x_train,axis=-1)
x_val=np.expand_dims(x_val,axis=-1)
x_test=np.expand_dims(x_test,axis=-1)

# #%97 training accuracy
model=tf.keras.models.Sequential([
    
    tf.keras.layers.InputLayer(input_shape=(x_train.shape[1],x_train.shape[2])),
    
    tf.keras.layers.Conv1D(64,kernel_size=3,strides=1,activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv1D(64,kernel_size=3,strides=2,activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv1D(128,kernel_size=5,strides=1,activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv1D(128,kernel_size=5,strides=1,activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
                
    tf.keras.layers.Conv1D(128,kernel_size=5,strides=2,activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
        
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,return_sequences=True)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv1D(256,kernel_size=5,strides=1,activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv1D(256,kernel_size=5,strides=1,activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv1D(256,kernel_size=5,strides=2,activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256,return_sequences=True)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv1D(512,kernel_size=5,strides=1,activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv1D(512,kernel_size=5,strides=1,activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv1D(512,kernel_size=5,strides=2,activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(256,activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512,activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1024,activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3,activation='softmax')])



# model=tf.keras.models.Sequential([

#     tf.keras.layers.InputLayer(input_shape=(x_train.shape[1],x_train.shape[2])),
    
#     tf.keras.layers.Conv1D(64,kernel_size=3,strides=1,activation='relu',padding='same'),
#     tf.keras.layers.Conv1D(64,kernel_size=3,strides=2,activation='relu',padding='same'),
#     tf.keras.layers.LSTM(64,return_sequences=True),
#     tf.keras.layers.BatchNormalization(),
    
#     tf.keras.layers.Conv1D(128,kernel_size=5,strides=1,activation='relu',padding='same'),
#     tf.keras.layers.Conv1D(128,kernel_size=5,strides=2,activation='relu',padding='same'),
#     tf.keras.layers.LSTM(128,return_sequences=True),
#     tf.keras.layers.BatchNormalization(),

#     tf.keras.layers.GlobalAveragePooling1D(),
#     tf.keras.layers.Dense(128,activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.01)),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(256,activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.01)),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(3,activation='softmax')])



model.summary()

f_name=datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=3))).strftime("%d %B %Y - %H:%M:%S")
log_dir = "logs/fit/" + f_name
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
file_writer = tf.summary.create_file_writer(log_dir+"/train")

delayed_lr_schedule = DelayedExponentialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=1,
    decay_rate=learning_rate_decay_factor,
    start_epoch=start_epoch,
    log_dir=log_dir)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['acc'])
start_time = datetime.datetime.now()

my_callback = [tensorboard_callback, delayed_lr_schedule,checkpoint_callback]
# my_callback = [tensorboard_callback, delayed_lr_schedule, early_stopping, model_checkpoint]

# Training model
history=model.fit(x=x_train,
          y=y_train,
          batch_size=batchSize,
          epochs=epochs,
          validation_data=(x_val,y_val),
          #callbacks=[tensorboard_callback,delayed_lr_schedule],
          callbacks = my_callback,
          verbose=1,
            )



end_time = datetime.datetime.now()
duration = end_time - start_time
hours, remainder = divmod(duration.seconds, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Training duration: {hours} hours, {minutes} minutes, {seconds} seconds")


# Confusion matrix and F1 score
model = load_model(checkpoint_path)

# start_time = time.time()
# y_pred=model.predict(x_test[0])
# end_time = time.time()
# elapsed = end_time-start_time
# print(elapsed)

y_pred=model.predict(x_test)
y_pred_str=cat.inverse_transform(y_pred)
y_test_str=cat.inverse_transform(y_test)

y_pred=np.argmax(y_pred, axis=1)
y_test=np.argmax(y_test, axis=1)

fig,axs=plt.subplots(1,2)
clf_report=classification_report(y_test,y_pred,target_names=np.unique(y_test_str),output_dict=True)
sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :4].T, annot=True,cmap='viridis',ax=axs[1])
cm = confusion_matrix(y_test,y_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=np.unique(y_test_str))
disp.plot(ax=axs[0])
plt.gcf().set_size_inches(16, 9)

plt.savefig(buffer,format='png')
buffer.seek(0)
image = tf.image.decode_png(buffer.getvalue(), channels=4)
image = tf.expand_dims(image, 0)
plt.show()


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(acc_buffer,format='png')
plt.show()
acc_buffer.seek(0)
acc_img = tf.image.decode_png(acc_buffer.getvalue(), channels=4)
acc_img = tf.expand_dims(acc_img, 0)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(loss_buffer,format='png')
plt.show()

loss_buffer.seek(0)
loss_img = tf.image.decode_png(loss_buffer.getvalue(), channels=4)
loss_img = tf.expand_dims(loss_img, 0)

save_evaluated_lc_plots(x_test,y_test_str,y_pred_str,buffer2)
buffer2.seek(0)
image2 = tf.image.decode_png(buffer2.getvalue(), channels=4)
image2 = tf.expand_dims(image2, 0)

with file_writer.as_default():
    tf.summary.image("Confusion Matrix", image, step=0)
    tf.summary.image("Evaluated Light Curve Plots", image2, step=0)
    tf.summary.image("Accuracy_legend", acc_img, step=0)
    tf.summary.image("Loss_legend", loss_img, step=0)
    tf.summary.text("Model",get_summary_str(model),step=0)
    tf.summary.text("Training Duration",f"Training duration: {hours} hours, {minutes} minutes, {seconds} seconds",step=0)

