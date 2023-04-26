from typing import Any
import argparse
import yaml
import keras.callbacks
from model_builder import ScalogramSegmentationLSTMModelBuilder
from dataset import ScalogramMatrixDataset
from dataset_builder import create_dataset
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        default=None,
        help='path to the data config file.'
    )
    parser.add_argument(
        '-e', '--epochs',
        default=20,
        type=int,
        help='number of epochs to train for.'
    )
    parser.add_argument(
        '-b', '--batch-size',
        dest='batch_size',
        default=32,
        type=int,
        help='batch size to load the data.'
    )
    parser.add_argument(
        '-pn', '--project-name',
        default=None,
        type=str,
        dest='project_name',
        help='training result dir name in outputs/training/, (default res_#).'
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="shuffle data before creating the Datasets."
    )
    parser.add_argument(
        "--valid-perc",
        type=float,
        dest="valid_perc",
        default=None,
        help="If provided, split the data into train and validation set, according to the given percentage."
    )
    args = vars(parser.parse_args())
    return args



def main(args : dict[str, Any]):
    device_name = tf.test.gpu_device_name()
    if device_name != "/device:GPU:0":
        raise RuntimeError("Not found GPU")

    NUM_EPOCHS = args["epochs"]
    CONFIG_FILE = args["config"]
    BATCH_SIZE = args["batch_size"]
    SHUFFLE = args["shuffle"]
    VALID_PERC = args["valid_perc"]
    SPLIT = VALID_PERC != None

    with open(CONFIG_FILE, "r") as f:
        configs = yaml.safe_load(f)

    DATA_PATH = configs["DATA_PATH"]
    TEST_PATH = configs["TEST_PATH"]
    WINDOW_SIZE = configs["WINDOW_SIZE"]
    DWT_LEVELS = configs["DWT_LEVELS"]
    
    test_data, _ = create_dataset(path=TEST_PATH, batch_size=BATCH_SIZE, window_size=WINDOW_SIZE, dwt_levels=DWT_LEVELS, shuffle=False, split=False)

    with tf.device('/device:GPU:0'):
        train_dataset, valid_dataset = create_dataset(DATA_PATH, BATCH_SIZE, WINDOW_SIZE, DWT_LEVELS, SHUFFLE, SPLIT, VALID_PERC)
        print("Dataset loaded")
        print("len train:", len(train_dataset))
        print("len vaid:", len(valid_dataset))
        builder = ScalogramSegmentationLSTMModelBuilder(WINDOW_SIZE, DWT_LEVELS)
        
        model = builder.build_network()
        print("Model built")
        print(model.summary())

        callback_list = [
            keras.callbacks.EarlyStopping(monitor = "val_loss", patience=2),
            keras.callbacks.ModelCheckpoint("outputs/model/model-{epoch:02d}.hdf5", verbose = 2, monitor ="accuracy")
        ]

        model.fit(x=train_dataset, validation_data=valid_dataset, epochs=NUM_EPOCHS, verbose=2, callbacks=callback_list)

        #Stampa un file di testo contenente la inference calcolata su dei dati di test
        with open("outputs/inference/results.txt", "w") as f:
            for num_batch in range(len(test_data)):
                chunks, labels = test_data[num_batch]

                predictions = model(chunks)
                for p in predictions:
                    f.write(f"{p}\n\n")
                    for i, elem in enumerate(p):
                        if elem > 0:
                            f.write(f"{i} -- {elem}\n")



if __name__ == "__main__":
    main(parse_args())