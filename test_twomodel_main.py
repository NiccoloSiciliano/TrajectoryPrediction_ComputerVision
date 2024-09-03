from Model import Model, TwoModelArch
from utils import Utils
from CellMask import CellMask
from BinaryMask import BinaryMask
from Main import Main

import tensorflow as tf
import cv2
import sys
import numpy as np
from var import n_classes, batch_size

# thing_to_classify wall_model_path nucleus_model_path dataset_path

if __name__ == "__main__":
	wall_model_path = sys.argv[2]
	nucleus_model_path = sys.argv[3]
	test_path = sys.argv[4]

	print()
	print("Accuracy per pixel: ",Main.test_twomodel_seg(wall_model_path, nucleus_model_path, test_path))