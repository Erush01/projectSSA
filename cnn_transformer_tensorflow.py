import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
from mmt import MiniMegaTortoraDataset
from ssaUtils import pad_to_size_interpolate
import argparse
class CNNFeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, base_filters=32, kernel_sizes=[5, 3, 3], embed_size=128, **kwargs):
        super(CNNFeatureExtractor, self).__init__(**kwargs)
        self.base_filters = base_filters
        self.kernel_sizes = kernel_sizes
        self.embed_size = embed_size
        
        self.cnn_layers = []
        current_filters = base_filters
        
        for k_size in kernel_sizes:
            layer = tf.keras.Sequential([
                tf.keras.layers.Conv1D(
                    filters=current_filters, 
                    kernel_size=k_size, 
                    padding='same'
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('gelu'),
                tf.keras.layers.Dropout(0.1)
            ])
            self.cnn_layers.append(layer)
            current_filters *= 2
        
        # Add a final projection layer to match embed_size
        self.projection = tf.keras.layers.Dense(embed_size)
    
    def call(self, inputs, training=False):
        x = inputs
        for layer in self.cnn_layers:
            x = layer(x, training=training)
        
        # Project to embed_size
        x = self.projection(x)
        return x

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_size, heads, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"
        
        # Linear projections
        self.query_dense = tf.keras.layers.Dense(embed_size)
        self.key_dense = tf.keras.layers.Dense(embed_size)
        self.value_dense = tf.keras.layers.Dense(embed_size)
        self.dense = tf.keras.layers.Dense(embed_size)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        
        # Linear transformations
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        # Split into multiple heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        
        # Softmax attention weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        # Attention output
        output = tf.matmul(attention_weights, value)
        
        # Reshape and linear projection
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.embed_size))
        
        return self.dense(output)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_size, heads, forward_expansion, dropout=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.feed_forward = tf.keras.Sequential([
            tf.keras.layers.Dense(forward_expansion * embed_size, activation='relu'),
            tf.keras.layers.Dense(embed_size)
        ])
        
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
    
    def call(self, inputs, training=False):
        # Self-attention
        attn_output = self.attention(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        
        # First residual and normalization
        out1 = self.norm1(inputs + attn_output)
        
        # Feed forward
        ffn_output = self.feed_forward(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        # Second residual and normalization
        return self.norm2(out1 + ffn_output)

class TimeSeriesTransformer(tf.keras.Model):
    def __init__(
        self, 
        feature_size, 
        num_classes, 
        embed_size=128, 
        num_layers=3,
        forward_expansion=2, 
        heads=4, 
        dropout=0.1,
        cnn_base_filters=32,
        cnn_kernel_sizes=[7,5,3,3],
        max_length=1000,
        **kwargs
    ):
        super(TimeSeriesTransformer, self).__init__(**kwargs)
        
        # CNN Feature Extractor
        self.feature_extractor = CNNFeatureExtractor(
            base_filters=cnn_base_filters, 
            kernel_sizes=cnn_kernel_sizes,
            embed_size=embed_size
        )
        
        # Positional Embedding
        self.position_embedding = tf.keras.layers.Embedding(max_length, embed_size)
        
        # Transformer Encoder Layers
        self.encoder_layers = [
            TransformerBlock(
                embed_size, 
                heads, 
                forward_expansion, 
                dropout
            ) for _ in range(num_layers)
        ]
        
        # Classification Head
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(embed_size, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    
    def call(self, inputs, training=False):
        # Extract features with CNN
        x = self.feature_extractor(inputs, training=training)
        
        # Add positional embedding
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = tf.expand_dims(positions, axis=0)
        positions = tf.repeat(positions, repeats=tf.shape(x)[0], axis=0)
        
        pos_emb = self.position_embedding(positions)
        x = x + pos_emb
        
        # Apply transformer layers
        for layer in self.encoder_layers:
            x = layer(x, training=training)
        
        # Global average pooling and classification
        x = tf.reduce_mean(x, axis=1)
        return self.classifier(x)

def train_tensorflow_transformer():
    # Similar to PyTorch version, but adapted for TensorFlow
    device = 'cuda' if tf.config.list_physical_devices('GPU') else 'cpu'
    parser=argparse.ArgumentParser(
        prog="SSA Classifier",
    )
    parser.add_argument('-d','--debug',action="store_true")
    args=parser.parse_args()

    if(args.debug): 
        satelliteNumber=[1,1,5]
    else:
        satelliteNumber=[60,160,300]

    # Hyperparameters
    trackSize = 500
    EPOCHS = 100
    batchSize = 32
    
    # Load Dataset
    mmt = MiniMegaTortoraDataset(satNumber=satelliteNumber, periodic=True)
    x, y = mmt.load_data_new()
    
    # Preprocessing
    x = [pad_to_size_interpolate(array, trackSize) for array in x]
    x = np.array(x)
    y = np.array(y)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = to_categorical(y)
    
    # Train-Val-Test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_train
    )
    
    # Normalization
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train.reshape(-1, trackSize)).reshape(-1, trackSize, 1)
    x_val = scaler.transform(x_val.reshape(-1, trackSize)).reshape(-1, trackSize, 1)
    x_test = scaler.transform(x_test.reshape(-1, trackSize)).reshape(-1, trackSize, 1)
    
    # Model
    model = TimeSeriesTransformer(
        feature_size=1,
        num_classes=3,
        embed_size=128,
        num_layers=3,
        forward_expansion=2,
        heads=8,
        cnn_base_filters=32,
        max_length=trackSize,
        cnn_kernel_sizes=[7,5,3,3],
        dropout=0.1
    )
    
    # Build the model by calling it with dummy input
   
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Learning Rate Scheduler
    # lr_scheduler = ReduceLROnPlateau(
    #     monitor='val_loss', 
    #     factor=0.1, 
    #     patience=5, 
    #     min_lr=1e-5
    # )
    
    # Training
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=batchSize,
        verbose=1
    )
    
    # Evaluation
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    # Plotting
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model

if __name__ == "__main__":
    train_tensorflow_transformer()