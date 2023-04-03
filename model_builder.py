from keras.layers import (
    Input,
    LSTM
)
from keras.losses import BinaryCrossentropy
from keras.models import Model


class ScalogramSegmentationLSTMModelBuilder:
    def __init__(
        self,
        window_size: int,
        dwt_levels : int,
    ):
        self.window_size = window_size
        self.dwt_levels = dwt_levels

    def build_network(self):
        inputs = Input(shape=(self.window_size, self.dwt_levels))

        lstm_out = LSTM(
                    units=1,
                    return_sequences=True,
                )(inputs)
        model = Model(inputs, lstm_out)

        model.compile(optimizer="adam", loss=BinaryCrossentropy())
        return model
if __name__ == "__main__":
    builder = ScalogramSegmentationLSTMModelBuilder(1000, 15)
    m = builder.build_network()
    print(m.summary())