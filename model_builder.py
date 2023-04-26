from keras.layers import (
    Input,
    LSTM,
    Dense,
    Activation
)
from keras.losses import BinaryCrossentropy, losses_utils
from keras.models import Model
from keras.activations import hard_sigmoid
import keras.backend as K
from keras.metrics import BinaryAccuracy
from keras.optimizers import Adam
import tensorflow as tf

#Non funziona, ha senso questa robaccia?
def Custom_Hamming_Loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    # batchsize, 1000, 1 -> batcsize, 1

    one_true = K.sum(y_true, axis=1)
    one_pred = K.sum(y_pred, axis=1)

    abs_v = K.abs(one_true - one_pred)
    return abs_v
    # return K.mean(y_true*(1-y_pred)+(1-y_true)*y_pred, axis=1)


class ScalogramSegmentationLSTMModelBuilder:
    def __init__(
        self,
        window_size: int,
        dwt_levels: int,
    ):
        self.window_size = window_size
        self.dwt_levels = dwt_levels

    def build_network(self):
        inputs = Input(shape=(self.window_size, self.dwt_levels))

        lstm_out1 = LSTM(
            units=128,
            return_sequences=True,
        )(inputs)
    

        outputs = Dense(
            units=1,
            activation=hard_sigmoid,
        )(lstm_out1)
        
        model = Model(inputs, outputs)

        acc = BinaryAccuracy(
            name="binary_accuracy", dtype=None, threshold=0.5)
        
        optim = Adam(learning_rate=0.01)

        model.compile(optimizer=optim,
                      loss=Custom_Hamming_Loss,
                      metrics=[acc])
        return model


if __name__ == "__main__":
    builder = ScalogramSegmentationLSTMModelBuilder(1000, 15)
    m = builder.build_network()
    print(m.summary())
