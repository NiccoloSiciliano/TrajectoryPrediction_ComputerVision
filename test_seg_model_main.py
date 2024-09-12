import sys
from Main import Main
from var import n_classes, clas
# clas dataset_path model_path

if __name__ == "__main__":
	test_path = sys.argv[2]

	model_path = sys.argv[3]
	
	Main.test_model(test_path, model_path)

	
