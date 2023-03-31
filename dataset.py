import csv
import glob
import math
import os
import numpy as np
from tensorflow import keras
from xml.etree import ElementTree as et


class ScalogramMatrixDataset(keras.utils.Sequence):
    

    def __init__(self, 
                 batch_size,
                 window_size,
                 dwt_levels,
                 data_path):
        self.window_size = window_size
        self.dwt_levels = dwt_levels
        self.batch_size = batch_size
        self.data_path = data_path
        
        self.annot_files = []
        self.all_windows : list[tuple[str, int]] = []

        self.current_annot_file_index = 0
        self.current_matrix = None
        self.curren_timestamp = 0
        self.block_ends = None

        self.read_data_directory()


    def read_data_directory(self):
        all_annotations = glob.glob(os.path.join(self.data_path, "*.xml"))
        invalid_xml = 0

        def check_path(path : str):
            global invalid_xml
            try:
                tree = et.parse(path)
            except:
                invalid_xml += 1
                return False
            
            root = tree.getroot()
            image_name = root.findtext("filename")
            image_path = os.path.join(self.images_path, image_name)
            discard_path = False
            if not os.path.exists(image_path):
                print(f"Image {image_path} associated to {path} not found...")
                print(f"Discarding {path}...")
                discard_path = True

            return not discard_path

        #Filtra gli xml non correttamente formattati e quelli che non hanno l'immagine corrispondente (capiterà?)
        self.annot_files = list(filter(check_path, all_annotations))
        print(f"XML INVALIDI: {invalid_xml}")

        #Per ogni annotation, vediti la size della matrice per determinare la quantità di windows prodotte da quella matrice
        for annot_file in self.annot_files:
            tree = et.parse(annot_file)
            root = tree.getroot()
            width = int(root.find("size").findtext("width"))
            if width < self.window_size:
                raise ValueError(f"Width {width} of file {annot_file} is less than window_size {self.window_size}")
            
            for offset in range(width - self.window_size):
                self.all_windows.append((annot_file, offset))




    def read_matrix_file(self, path : str):
        reader = csv.reader(open(path, "r"), delimiter=",")
        x = list(reader)
        result = np.array([x]).astype(np.float32)
        
        return result

    def load_matrix_and_label(self, index: int):
        annot_filepath = self.annot_files[index]

        tree = et.parse(annot_filepath)
        root = tree.getroot()
        matrix_name :str = root.findtext("filename")
        matrix_path :str = os.path.join(self.data_path, matrix_name)
        matrix_width = root.find("size").findtext("width")

        matrix = self.read_matrix_file(matrix_path)
        
        block_ends = set()

        for member in root.find("boxes").findall("object"):
            end_x = member.find("bndbox").find("xmax").text
            #Non c'è la fine di un blocco alla fine di una matrice
            if end_x != matrix_width:
                block_ends.add(end_x)

        return matrix, block_ends
        

    def create_batch_fragments(self, matrix : np.ndarray):
        pass


    def __len__(self) -> int:
        return 0

    def __getitem__(self, index : int):
        
        if self.current_matrix is None:
            self.current_matrix, self.block_ends = self.load_matrix_and_label(self.current_annot_file_index)
            self.current_annot_file_index += 1
            