
from Main import Main
import sys
import os

if __name__ == "__main__":
	image_path = sys.argv[1]
	model_path = sys.argv[2]
	count = 0
	matches = 0
	clas_map = {0:0,1:0,2:0,3:0,4:0}
	for i, d in enumerate(["../GrThClasses/Test/Basophil/","../GrThClasses/Test/Eosinophil/","../GrThClasses/Test/Lymphocyte/","../GrThClasses/Test/Monocyte/","../GrThClasses/Test/Neutrophil/"]):
		for e in os.listdir(d):
			pred = Main.test_classification_image(model_path, d+e)
			if pred == i:
				matches += 1
				clas_map[pred] += 1
			count += 1
	print(matches/count)
	print(clas_map)