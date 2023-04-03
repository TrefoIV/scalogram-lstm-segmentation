import csv
import glob
import math
import os
import numpy as np
from tensorflow import keras
from xml.etree import ElementTree as et


class ScalogramMatrixDataset(keras.utils.Sequence):
    

    def __init__(self, 
                 batch_size : int,
                 window_size : int,
                 dwt_levels : int,
                 data_path : str,
                 data_description : list[tuple[str, int]]):
        self.window_size = window_size
        self.dwt_levels = dwt_levels
        self.batch_size = batch_size
        self.data_path = data_path
        
        self.all_windows : list[tuple[str, int]] = data_description


    def read_matrix_chunk_and_prepare_labels(self, path : str, offsets: list[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        '''
        Read all specified matrix chunks. Returns a dictionary of int -> tuple[chunk, labels].
        Each key is a specific offset, and the corresponding value is a tuple with the matrix chunk and
        an empty label vector.
        Each chunk is TRANSPOSED, so it has shape (window_size, dwt_levels) and represents a vector of timestamps,
        each containing the column vector of the matrix.
        '''
        reader = csv.reader(open(path, "r"), delimiter=",")
        x = list(reader)
        matrix = np.array(x).astype(np.float32)
        result = {}
        for offset, _ in offsets:
            chunk = matrix[:, offset : (offset+self.window_size)].transpose()
            empty_label = np.zeros((chunk.shape[0], 1))
            result[offset] = (chunk, empty_label)

        return result

    def load_chunks_and_labels(self, batch : list[tuple[str, list[tuple[int, int]]]]) -> tuple[np.ndarray, np.ndarray]:
        '''
        Load a batch of matrix chunks with corresponding labels, in the same order as in the batch input list.
        Returns a tuple: the first element is an array of shape (batch_size, window_size, dwt_levels) with all the matrix_chunks;
        the second element is an array of shape (batch_size, )
        ''' 
        #Group all offsets chunks of the same matrix -> open the file one time only
        groups = {x : [(y[1], index) for index,y in enumerate(batch) if y[0] == x ] for x in set(map(lambda v : v[0] ,batch))}

        chunks = np.zeros((len(batch), self.window_size, self.dwt_levels))
        labels = np.zeros((len(batch), self.window_size, 1))

        for annot_filepath, offsets in groups.items():
            tree = et.parse(annot_filepath)
            root = tree.getroot()
            matrix_name :str = root.findtext("filename")
            matrix_path :str = os.path.join(self.data_path, matrix_name)
            matrix_width = root.find("size").findtext("width")

            chunks_with_labels = self.read_matrix_chunk_and_prepare_labels(matrix_path, offsets)
            
            block_ends = set()

            for member in root.find("boxes").findall("object"):
                end_x = int(member.find("bndbox").find("xmax").text)
                #Non c'è la fine di un blocco alla fine di una matrice
                if end_x != matrix_width:
                    block_ends.add(end_x)
            
            for offset, index in offsets:
                for end in block_ends:
                    if end > offset and end < offset + self.window_size:
                        chunks_with_labels[offset][1][end - offset] = 1
                chunks[index] = chunks_with_labels[offset][0]
                labels[index] = chunks_with_labels[offset][1]
            
        return chunks, labels
        


    def __len__(self) -> int:
        return math.ceil(len(self.all_windows)/self.batch_size)

    def __getitem__(self, index : int):
        
        batch_start = index * self.batch_size
        batch_end = min((index+1)*self.batch_size, len(self.all_windows))
        batch = self.all_windows[batch_start : batch_end]

        return self.load_chunks_and_labels(batch)