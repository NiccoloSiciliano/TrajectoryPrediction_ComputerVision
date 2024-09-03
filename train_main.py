import sys
from Main import Main
from var import n_classes, clas


if __name__ == "__main__":
	path_dtset = sys.argv[2]

	train_path = path_dtset+"/Train/"
	val_path = path_dtset+"/Validation/"
	test_path = path_dtset+"/Test/"
	working = "./"

	Main.train_model(train_path, val_path, working)

	
