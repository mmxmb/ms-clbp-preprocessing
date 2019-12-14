from multiprocessing import Pool, cpu_count
import os
import cv2
import pickle
import numpy as np
from typing import  NewType
from ms_clbp import ms_clbp_feature_matrices

Matrix = NewType('Matrix', np.ndarray)

def normalize(mat: Matrix, new_min: float = 0, new_max: float = 1) -> Matrix:
    _min, _max = np.amin(mat), np.amax(mat)
    return np.multiply(mat - _min, (new_max - new_min) / (_max - _min)) + new_min

def preprocess(dataset_path, classes, args, target_path):
    """ Preprocess dataset using MS-CLBP technique.
        `dataset_path` is a path to the directory with dataset
            classes, each class in its own directory
        `classes` is a list of class names (also directory names)
            in `dataset_path`
        `args` is a list containing arguments for ms_clbp_feature_matrices,
            except for the first `img` argument
        `target_path` is a path where pickled MS-CLBP feature tensors
            are saved; one pickle per class
    """
    for cls_num, cls in enumerate(classes):
        class_dir = os.path.join(dataset_path, cls)
        arguments = []
        for filename in sorted(os.listdir(class_dir)):
            filepath = os.path.join(class_dir, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img.shape != (256, 256):
                img = cv2.resize(img, (256, 256))
            img = normalize(img)
            a = [img] + args
            arguments.append(a)
        with Pool(cpu_count()) as p:
            feature_vecs = p.starmap(ms_clbp_feature_matrices, arguments)
        with open(os.path.join(os.getcwd(), target_path, f"{cls}_feature_mats.pkl"), 'wb') as f:
            pickle.dump(feature_vecs, f)

if __name__ == "__main__":
    scales = [1, 1/2, 1/3, 1/4]
    n_points = 8
    radii = [x for x in range(1, 7)]
    patch_size = 32
    n_bins = 16
    args = [scales, n_points, radii, patch_size, n_bins]

    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset')

    classes = ['forest', 'buildings', 'river', 'mobilehomepark', 'harbor', 'golfcourse', 'agricultural', 'runway', 'baseballdiamond', 'overpass', 'chaparral', 'tenniscourt', 'intersection', 'airplane', 'parkinglot', 'sparseresidential', 'mediumresidential', 'denseresidential', 'beach', 'freeway', 'storagetanks']
    classes.sort()

    target_path = os.path.join(os.path.dirname(__file__), '..', 'pkl')


    preprocess(dataset_path, classes, args.copy(), target_path)
