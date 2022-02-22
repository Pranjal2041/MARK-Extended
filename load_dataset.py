# Sample file on how to load the datasets

import pickle
from data_utils import ImageGridData, ImageData, MyDS

dset_grid = pickle.load(open('datasets/image_grid_data.pickle', 'rb'))
dset_image = pickle.load(open('datasets/image_data.pickle', 'rb'))