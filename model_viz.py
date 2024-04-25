import visualkeras
import tensorflow as tf
from PIL import ImageFont
from collections import defaultdict

color_map = defaultdict(dict)
color_map[tf.keras.layers.Conv1D]['fill'] = '#512052'

model=tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(1000,64)),
    tf.keras.layers.Conv1D(64,2,activation='relu'),
    tf.keras.layers.MaxPool1D(pool_size=(2)),
    tf.keras.layers.LSTM(64,return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(128,2,activation='relu'),
    tf.keras.layers.MaxPool1D(pool_size=(2)),
    tf.keras.layers.LSTM(128,return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3,activation='softmax')])


font = ImageFont.truetype("arial.ttf", 12)
visualkeras.layered_view(model, legend=True, draw_volume=True,font=font,color_map=color_map,scale_xy=0.8, scale_z=0.2,max_xy=100,to_file='model_viz.png').show()# selected font
