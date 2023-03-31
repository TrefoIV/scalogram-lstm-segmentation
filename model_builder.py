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


class TextSegmentationModelBuilder:
    def __init__(
        self,
        embedding_layer: Layer,
        n_sentencewise_lstm_layers: int,
        n_documentwise_lstm_layers: int,
        lstm_dimension: int,
    ):
        """
        :param embedding_layer: Keras embedding layer. 
            The layer is passed as argument, because the merge of pre-trained and custom vocab 
            for building the embedding matrix is managed by the training script.
        :param n_sentencewise_lstm_layers: Number of LSTM layers for the word-level network
        :param n_documentwise_lstm_layers: Number of LSTM layers for the sentence-level network
        :param lstm_dimension: Dimension (units) of LSTM layers
        """


        self.embedding_layer = embedding_layer
        self.lstm_dimension = lstm_dimension
        self.n_sentencewise_lstm_layers = n_sentencewise_lstm_layers
        self.n_documentwise_lstm_layers = n_documentwise_lstm_layers

    def build_network(self):
        sentences = Input(shape=(None, None))

        x = self.embedding_layer(sentences)

        # Word-level LSTM
        for layer in range(self.n_sentencewise_lstm_layers):
            x = TimeDistributed(
                Bidirectional(LSTM(
                    units=self.lstm_dimension,
                    return_sequences=True
                ))
            )(x)

        x = TimeDistributed(GlobalMaxPool1D())(x)

        # Sentence-level LSTM
        for layer in range(self.n_documentwise_lstm_layers):
            x = Bidirectional(LSTM(units=self.lstm_dimension, return_sequences=True))(x)

        x = TimeDistributed(Dense(1, activation="sigmoid"))(x)

        model = Model(sentences, x)
        return model