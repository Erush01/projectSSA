import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
from ssaUtils import get_summary_str,train_table,DiscreteWaveletTransform,DelayedExponentialDecay,save_evaluated_lc_plots,pad_to_size_interpolate,DiscreteWaveletTransform1
import argparse

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model // num_heads
        )
        
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training=False):
        # Multi-head self-attention
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=700):
        super(PositionalEncoding, self).__init__()
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pos_encoding = tf.constant(pe, dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:x.shape[1], :]

def create_hybrid_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    
    x = tf.keras.layers.Reshape((input_shape[0], input_shape[1]))(inputs)

    # Initial Convolutional Block
    x = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Conv1D(128, kernel_size=5, strides=2, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Conv1D(256, kernel_size=5, strides=2, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Positional Encoding for Transformer
    x = PositionalEncoding(d_model=x.shape[-1])(x)
    
    # Transformer Encoder Layers
    for _ in range(3):
        x = TransformerEncoderLayer(
            d_model=x.shape[-1], 
            num_heads=4, 
            dff=256
        )(x, training=True)
    
    # Bidirectional LSTM for additional sequence understanding
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(256, return_sequences=True)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Fully Connected Layers with Regularization
    x = tf.keras.layers.Dense(512, activation='relu', 
                               kernel_regularizer=tf.keras.regularizers.L2(0.01))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(256, activation='relu', 
                               kernel_regularizer=tf.keras.regularizers.L2(0.01))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Output Layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model




if __name__=="__main__":
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
    epochs = 100     # Number of epochs for training
    batchSize = 32        # batch size for training
    buffer = BytesIO()
    buffer2=BytesIO()
    acc_buffer=BytesIO()
    loss_buffer=BytesIO()


    #--------------------Learning Rate Scheduler-------------------------------

    learning_rate=1e-4
    final_learning_rate = 5e-6
    start_epoch=70
    learning_rate_decay_factor = (final_learning_rate /learning_rate)**(1/(epochs-start_epoch))

    #-------------------------------------------------------

    mmt=MiniMegaTortoraDataset(satNumber=satelliteNumber,periodic=True)

    x,y=mmt.load_data_new()
    classes=[[x] for x in np.unique(y)]


    #x=DiscreteWaveletTransform1(x)

    x = DiscreteWaveletTransform(x,wavelet='db20')
    x=[pad_to_size_interpolate(array,trackSize) for array in x]
    # x=[np.split(y,4) for y in x] # Only use if using ConvLSTM


    #Numpy array conversion        
    x=np.array(x)
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

    print_rich(train_table(x_train,y_train,x_val,y_val,x_test,y_test))

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

    # In your main training script, replace the model creation with:
    model = create_hybrid_model(
        input_shape=(x_train.shape[1], x_train.shape[2], 1), 
        num_classes=3
    )
    model.summary()

    # Compile and train as before
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['acc']
    )
    start_time = datetime.datetime.now()

    # Training remains the same
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batchSize,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[tensorboard_callback, delayed_lr_schedule],
        verbose=1
    )
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training duration: {hours} hours, {minutes} minutes, {seconds} seconds")

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

