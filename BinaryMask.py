import tensorflow as tf
import numpy as np
tf.config.run_functions_eagerly(True)

class BinaryMask:
    def __init__(self, binary_mask):
        self.binary_mask = binary_mask

    def get_masked_image(self,image):
        # Get the indices of the maximum value along the last axis (channel axis)
        v = tf.argmax(self.binary_mask, axis=-1)

        # Create a boolean mask where the value is 1
        mask = tf.equal(v, 1)

        # Use the mask to select pixels from the original image
        masked_image = tf.where(mask[...,tf.newaxis], image, tf.zeros_like(image))

        # Convert to uint8 for image representation
        masked_image = tf.cast(masked_image, tf.float32)

        return masked_image

    def get_bin_mask_image(self):
        binary_mask = self.get_binary_mask()
        h, w ,_ = binary_mask.shape
        new_mask = []
        for i in range(h):
            new_mask.append([])
            for j in range(w):
                clas = np.argmax(binary_mask[i][j])

                v = 0
                if clas == 1:
                    v = 255
                new_mask[i].append(v)

        return new_mask

    def get_three_mask_image(self):
        binary_mask = self.get_binary_mask()
        h, w ,_ = binary_mask.shape
        new_mask = []
        v = [0 , 120, 255]
        for i in range(h):
            new_mask.append([])
            for j in range(w):
                clas = np.argmax(binary_mask[i][j])
                new_mask[i].append(v[clas])
        return new_mask

    def get_binary_mask(self):
        return self.binary_mask
