
from Main import Main
import sys
import os
from utils import  Utils
if __name__ == "__main__":
	image_path = sys.argv[1]
	model_path = sys.argv[2]
	print(Main.test_classification_image(model_path, image_path))