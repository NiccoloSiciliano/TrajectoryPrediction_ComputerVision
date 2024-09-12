import sys

batch_size = 32
clas = sys.argv[1] if len(sys.argv) > 1 else "null"
n_classes = 3 if clas == "ALL" else 2

activation  = "softmax"
epochs = 1
image_size = 256