
import cv2
import psutil
import os


import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from var import activation, epochs, n_classes, batch_size
tf.config.run_functions_eagerly(True)

from BinaryMask import BinaryMask
from CellMask import CellMask
from Model import Model, ClassificationModel, TwoModelArch
from utils import Utils

class Main:
	def test_model(test_path, model_path):
		model = Model(model_path,n_classes)
		model.load_model()
		# Evaluate the model
		test_dataset = Utils.create_dataset(test_path+"images", test_path+"masks")
		test_dataset = test_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


		print()
		print("Accuracy on test: ", model.evaluate_model(test_dataset))

	def train_model(train_path, val_path, working):
		images_dir = train_path+'images'
		masks_dir = train_path+'masks'

		dataset = Utils.create_dataset(images_dir, masks_dir)
		val_dataset = Utils.create_dataset(val_path+"images", val_path+"masks")

		# Batch and shuffle the dataset
		dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
		val_dataset = val_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

		#Building Model
		model = Model("",n_classes)
		model.build_CNN()
		model.compile_model()
		print(n_classes, activation)
		model.train_model(dataset, val_dataset)
		count = epochs
		model.save(working+str(count)+"_wallarch_"+str(n_classes)+"_muldic_"+activation+"_all.h5")

	def test_image(model_path, image_path, mask_path):
		#Load models
		model = Model(model_path, 3)
		model.load_model()
		image, mask = Utils.load_image(image_path,mask_path)

		

		#Predict cell walls
		pred = model.feed(image)
		pred = BinaryMask (Utils.one_hot_vec(pred))

		cell_mask = np.array(pred.get_three_mask_image())

		original_mask = CellMask(mask_path)
		original_mask = (tf.cast(original_mask.get_mask_image(), tf.float32)).numpy()

		cell_mask_res = cv2.cvtColor(cell_mask.astype(np.uint8),cv2.COLOR_GRAY2BGR)
		original_mask_res = cv2.cvtColor(original_mask.astype(np.uint8),cv2.COLOR_GRAY2BGR)
		image_res = (image.numpy()*255).astype(np.uint8)

		result = np.hstack((image_res,original_mask_res, cell_mask_res))
		print("Accuracy Point: ",Utils.accuracy(mask.numpy(), pred.get_binary_mask()))
		cv2.imshow("One Model Architecture", result) 
		cv2.waitKey(0)

	def test_classification_model(model_path, test_path):
		#Load image
		model = ClassificationModel(model_path, 5)
		model.load_model()
		return model.evaluate_model(test_path)

	def train_classification_model(train_path, val_path, working):
		dataset = Utils.create_classification_dataset(train_path)
		classes_counts = dataset.classes
		unique, counts = np.unique(classes_counts, return_counts=True)
		print(dict(zip(unique, counts)))
		val_dataset = Utils.create_classification_dataset(val_path)
		print("Starting training")
		model = ClassificationModel("",n_classes)
		model.build_CNN()
		model.compile_model()
		print(n_classes, activation)
		model.train_model(dataset, val_dataset)
		model.save("classification_arch1_"+str(n_classes)+"_muldic_"+activation+"_all_classification.h5")

	def test_classification_image(model_path, image_path):
		#Load models
		model = ClassificationModel(model_path, 5)
		model.load_model()
		image = Utils.load_classification_image(image_path)

		

		#Predict cell walls
		pred = model.feed(image)
		return np.argmax(pred)

	def test_twomodel_seg(wall_model_path,nucleus_model_path, test_path):
		model = TwoModelArch(wall_model_path, nucleus_model_path,n_classes)
		model.load_model()
		# Evaluate the model
		test_dataset = Utils.create_dataset(test_path+"images", test_path+"masks")
		test_dataset = test_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
		return model.evaluate_model(test_dataset)