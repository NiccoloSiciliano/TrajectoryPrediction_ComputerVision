import sys
from Main import Main
from var import n_classes, clas
# clas dataset_path model_path

if __name__ == "__main__":
	path_dtset = sys.argv[2]

	test_path = path_dtset+"/Test/"
	model_path = sys.argv[3]
	
	Main.test_model(test_path, model_path)

	
