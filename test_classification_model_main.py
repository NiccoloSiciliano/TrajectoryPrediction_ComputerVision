
from Main import Main
import sys

if __name__ =="__main__":
	print("precision, recall, f1:",Main.test_classification_model(sys.argv[1],sys.argv[2]))