from keras.layers import (
    Layer,
    Dense,
    Input,
    LSTM,
    TimeDistributed,
    Bidirectional,
    GlobalMaxPool1D,
    BatchNormalization
)
from keras.models import Model
from keras.models import Sequential 


class ScalogramSegmentationLSTMModelBuilder:
    def __init__(
        self,
        window_size: int,
        dwt_levels : int,
    ):
        self.window_size = window_size
        self.dwt_levels = dwt_levels

    def build_network(self):
        scalogram_fragments = (self.window_size, self.dwt_levels)

        lstm_layer = LSTM(
                    units=1,
                    return_sequences=True,
                    input_shape = scalogram_fragments
                )
        model = Sequential()
        model.add(lstm_layer)
        return model

builder = ScalogramSegmentationLSTMModelBuilder(1000, 15)
m = builder.build_network()
print(m.summary())