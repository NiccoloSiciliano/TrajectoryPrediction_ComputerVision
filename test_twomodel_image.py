from Model import Model
from utils import Utils
from CellMask import CellMask
from BinaryMask import BinaryMask
from Main import Main

import tensorflow as tf
import cv2
import sys
import numpy as np

if __name__ == "__main__":
	image_path = sys.argv[2]
	mask_path = sys.argv[3]
	wall_model_path = sys.argv[4]
	nucleus_model_path = sys.argv[5]

	Main.test_twomodel_image(wall_model_path,nucleus_model_path, image_path, mask_path)