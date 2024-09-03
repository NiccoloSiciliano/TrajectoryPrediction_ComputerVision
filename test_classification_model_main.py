
from Main import Main
import sys

if __name__ =="__main__":
	print("Accuracy: ",Main.test_classification_model(sys.argv[1],sys.argv[2]))