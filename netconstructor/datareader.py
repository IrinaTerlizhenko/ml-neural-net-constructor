import os

import numpy as np
from matplotlib.image import imread


def load_from_csv(file, delimiter=','):
    return np.genfromtxt(file, dtype=np.float, delimiter=delimiter)


def load_from_img(file, rec: bool = False):
    if os.path.isfile(file):
        return np.array([imread(file), ])
    elif os.path.isdir(file):
        return _files_list(file, rec)


def _files_list(directory, rec: bool = False):
    files = []
    for r, d, f in os.walk(directory):
        for file in f:
            img = imread(os.path.join(r, file))
            files.append(img)
        if not rec:
            break

    return np.array(files)
