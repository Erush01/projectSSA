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

from tensorflow import keras
from keras import layers

class MBConvBlock(layers.Layer):
    """Mobile Inverted Bottleneck Convolution Block for 1D time series"""
    def __init__(self, filters, kernel_size=3, expand_ratio=4, se_ratio=0.25, 
                 strides=1, drop_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.strides = strides
        self.drop_rate = drop_rate
        
        # Expansion phase
        expanded_filters = int(filters * expand_ratio)
        self.expand_conv = layers.Conv1D(expanded_filters, 1, padding='same', use_bias=False)
        self.expand_bn = layers.BatchNormalization()
        self.expand_activation = layers.Activation('swish')
        
        # Depthwise convolution
        self.depthwise_conv = layers.DepthwiseConv1D(
            kernel_size, strides=strides, padding='same', use_bias=False
        )
        self.depthwise_bn = layers.BatchNormalization()
        self.depthwise_activation = layers.Activation('swish')
        
        # Squeeze and Excitation
        se_filters = max(1, int(filters * se_ratio))
        self.se_reduce = layers.Conv1D(se_filters, 1, activation='swish')
        self.se_expand = layers.Conv1D(expanded_filters, 1, activation='sigmoid')
        
        # Output projection
        self.project_conv = layers.Conv1D(filters, 1, padding='same', use_bias=False)
        self.project_bn = layers.BatchNormalization()
        
        # Dropout
        if drop_rate > 0:
            self.dropout = layers.Dropout(drop_rate)
        else:
            self.dropout = None
    
    def call(self, inputs, training=None):
        x = inputs
        
        # Expansion
        if self.expand_ratio != 1:
            x = self.expand_conv(x)
            x = self.expand_bn(x, training=training)
            x = self.expand_activation(x)
        
        # Depthwise
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x, training=training)
        x = self.depthwise_activation(x)
        
        # Squeeze and Excitation
        se = tf.reduce_mean(x, axis=1, keepdims=True)
        se = self.se_reduce(se)
        se = self.se_expand(se)
        x = x * se
        
        # Project
        x = self.project_conv(x)
        x = self.project_bn(x, training=training)
        
        # Residual connection and dropout
        if self.strides == 1 and inputs.shape[-1] == self.filters:
            if self.dropout is not None:
                x = self.dropout(x, training=training)
            x = x + inputs
        
        return x

class MultiHeadSelfAttention(layers.Layer):
    """Multi-head self-attention for 1D sequences"""
    def __init__(self, embed_dim, num_heads, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        self.head_dim = embed_dim // num_heads
        
        self.query_dense = layers.Dense(embed_dim, use_bias=False)
        self.key_dense = layers.Dense(embed_dim, use_bias=False)
        self.value_dense = layers.Dense(embed_dim, use_bias=False)
        self.combine_heads = layers.Dense(embed_dim)
        
        self.dropout = layers.Dropout(dropout)
        
    def attention(self, query, key, value, training=None):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(self.head_dim, tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        weights = self.dropout(weights, training=training)
        output = tf.matmul(weights, value)
        return output, weights
    
    def separate_heads(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        query = self.separate_heads(query)
        key = self.separate_heads(key)
        value = self.separate_heads(value)
        
        attention, weights = self.attention(query, key, value, training=training)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, seq_len, self.embed_dim))
        output = self.combine_heads(concat_attention)
        
        return output

class PositionalEncoding(layers.Layer):
    """Positional encoding layer for transformer"""
    def __init__(self, max_seq_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.pos_encoding = self.add_weight(
            name='pos_encoding',
            shape=(max_seq_len, embed_dim),
            initializer='zeros',
            trainable=True
        )
    
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:seq_len, :]
    """Transformer block with multi-head attention and feed-forward network"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='swish'),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TransformerBlock(layers.Layer):
    """Transformer block with multi-head attention and feed-forward network"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='swish'),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    """Transformer block with multi-head attention and feed-forward network"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='swish'),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def create_coatnet_timeseries(
    sequence_length,
    num_features,
    num_classes,
    conv_blocks_config=None,
    transformer_blocks=2,
    embed_dim=128,
    num_heads=8,
    ff_dim=256,
    dropout=0.1,
    pool_size=4
):
    """
    Create CoAtNet model for time series classification
    
    Args:
        sequence_length: Length of input time series
        num_features: Number of features per time step
        num_classes: Number of output classes
        conv_blocks_config: List of tuples (filters, num_blocks, strides) for conv stages
        transformer_blocks: Number of transformer blocks
        embed_dim: Embedding dimension for transformer (must be divisible by num_heads)
        num_heads: Number of attention heads
        ff_dim: Feed-forward dimension
        dropout: Dropout rate
        pool_size: Pooling size between conv and transformer stages
    
    Returns:
        Keras model
    """
    
    # Validate parameters
    if embed_dim % num_heads != 0:
        raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
    
    if conv_blocks_config is None:
        conv_blocks_config = [
            (64, 2, 1),   # Stage 0: 64 filters, 2 blocks, stride 1
            (128, 3, 2),  # Stage 1: 128 filters, 3 blocks, stride 2
            (256, 4, 2),  # Stage 2: 256 filters, 4 blocks, stride 2
        ]
    
    inputs = layers.Input(shape=(sequence_length, num_features))
    x = inputs
    
    # Initial convolution stem
    x = layers.Conv1D(32, 7, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    # Convolutional stages with MBConv blocks
    for stage_idx, (filters, num_blocks, strides) in enumerate(conv_blocks_config):
        for block_idx in range(num_blocks):
            block_strides = strides if block_idx == 0 else 1
            x = MBConvBlock(
                filters=filters,
                strides=block_strides,
                drop_rate=dropout * (stage_idx + 1) * 0.1,
                name=f'mbconv_stage{stage_idx}_block{block_idx}'
            )(x)
    
    # Transition from conv to transformer
    # Pooling to reduce sequence length
    x = layers.AveragePooling1D(pool_size=pool_size, padding='same')(x)
    
    # Always project to embedding dimension to ensure compatibility
    x = layers.Dense(embed_dim)(x)
    
    # Add positional encoding
    max_pos_len = sequence_length // pool_size + 10  # Add some buffer
    x = PositionalEncoding(max_pos_len, embed_dim,ff_dim)(x)
    
    # Transformer blocks
    for i in range(transformer_blocks):
        x = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            name=f'transformer_block_{i}'
        )(x)
    
    # Global average pooling and classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout)(x)
    
    # Classification head
    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid', name='predictions')(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = keras.Model(inputs, outputs, name='CoAtNet_TimeSeries')
    return model



physical_devices = tf.test.gpu_device_name()
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"


# Set the path where you want to save the best weights
checkpoint_path = "checkpoints/coatnet.keras"

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
start_epoch=10
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
"""
config1 = (32, 2, 1),   # Smaller for shorter sequences
            (64, 2, 2),
            (128, 3, 2),

config2 = (128, 2, 1),   # Smaller for shorter sequences
            (256, 2, 2),
            (512, 3, 2),

config3 = (32, 2, 1),   # Smaller for shorter sequences
            (64, 2, 2),
            (128, 3, 2),

"""

model = create_coatnet_timeseries(
        sequence_length=700,
        num_features=1,  # Single time series
        num_classes=3,
        conv_blocks_config=[
            (32, 2, 1),   # Smaller for shorter sequences
            (64, 2, 2),
            (128, 3, 2),
        ],
        transformer_blocks=2,
        embed_dim=185,  # Must be divisible by num_heads
        num_heads=5,    # 128/8 = 16 (valid)
        ff_dim=256,
        dropout=0.5,
        pool_size=4
    )

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
custom_objects = {
            'MBConvBlock': MBConvBlock,
            'MultiHeadSelfAttention': MultiHeadSelfAttention,
            'TransformerBlock': TransformerBlock,
            'PositionalEncoding': PositionalEncoding
        }


model = keras.models.load_model(checkpoint_path, custom_objects=custom_objects)

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

