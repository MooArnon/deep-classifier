import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from .custom_metric import BalancedLabelMetric

def build_dynamic_dnn_hp(hp, input_shape):
    inputs = layers.Input(shape=input_shape)

    x = inputs

    # Dynamically choose number of hidden layers (between 1 and 5 layers)
    num_layers = hp.Int('num_layers', min_value=1, max_value=6)

    for i in range(num_layers):
        units = hp.Int(f'units_layer_{i+1}', min_value=16, max_value=2048, step=16)
        x = layers.Dense(units, activation='relu')(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    # Dynamically choose learning rate
    learning_rate = hp.Float('learning_rate', min_value=1e-8, max_value=1e-2, sampling='log')

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[BalancedLabelMetric()]
    )

    return model
