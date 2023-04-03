import glob
import math
import os
from xml.etree import ElementTree as et
from random import shuffle
from dataset import ScalogramMatrixDataset

def _read_data_directory(data_path, window_size : int):
        all_annotations = glob.glob(os.path.join(data_path, "*.xml"))
        invalid_xml = 0
        all_windows = []

        def check_path(path : str):
            global invalid_xml
            try:
                tree = et.parse(path)
            except:
                invalid_xml += 1
                return False
            
            root = tree.getroot()
            image_name = root.findtext("filename")
            image_path = os.path.join(data_path, image_name)
            discard_path = False
            if not os.path.exists(image_path):
                print(f"Image {image_path} associated to {path} not found...")
                print(f"Discarding {path}...")
                discard_path = True

            return not discard_path

        #Filtra gli xml non correttamente formattati e quelli che non hanno l'immagine corrispondente (capiterà?)
        annot_files = list(filter(check_path, all_annotations))
        print(f"XML INVALIDI: {invalid_xml}")

        #Per ogni annotation, vediti la size della matrice per determinare la quantità di windows prodotte da quella matrice
        for annot_file in annot_files:
            tree = et.parse(annot_file)
            root = tree.getroot()
            width = int(root.find("size").findtext("width"))
            if width < window_size:
                raise ValueError(f"Width {width} of file {annot_file} is less than window_size {window_size}")
            
            for offset in range(width - window_size):
                 all_windows.append((annot_file, offset))


def create_dataset(path : str,
                   batch_size : int,
                   window_size : int,
                   dwt_levels : int,
                   shuffle : bool, 
                   split : bool, 
                   valid_perc : float = 0.15) -> tuple[ScalogramMatrixDataset, ScalogramMatrixDataset]:
    all_windows = _read_data_directory(path)
    if shuffle:
         shuffle(all_windows)
    train_data = all_windows
    valid_data = []

    if split:
        if valid_perc < 0 or valid_perc >= 1:
             raise ValueError(f"Split percentage must be a float from 0 to 1, got {valid_perc}")
        split_index = math.floor(len(all_windows) * valid_perc)
        train_data = all_windows[:split_index]
        valid_data = all_windows[split_index:]
    
    train_dataset = ScalogramMatrixDataset(batch_size, window_size, dwt_levels, train_data)
    valid_dataset = ScalogramMatrixDataset(batch_size, window_size, dwt_levels, valid_data)
    return train_dataset, valid_dataset