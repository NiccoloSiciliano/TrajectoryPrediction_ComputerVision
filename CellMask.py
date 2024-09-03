
import tensorflow as tf
from BinaryMask import BinaryMask

tf.config.run_functions_eagerly(True)
class CellMask:

    def __init__(self, mask_path):
        #Load mask
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, [256, 256],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask = tf.cast(mask, tf.uint8)    # Masks are often in integers
        self.mask = mask

    def get_cell_wall_mask(self):
        # Convert mask to TensorFlow tensor if it isn't already
        wall_mask = tf.cast(self.mask, dtype=tf.float32)

        # Create a binary mask where pixels with values >= 40 are marked as 1 (cell wall), else 0 (nothing)

        binary_mask_nothing = tf.where(wall_mask<= 20, 1, 0)

        binary_mask_wall = tf.where(wall_mask > 20, 1,0)

        # One-hot encode the binary mask
        wall_mask = tf.concat([binary_mask_nothing, binary_mask_wall], axis=2)

        return BinaryMask(wall_mask)

    def get_nucleus_mask(self):
        # Convert mask to TensorFlow tensor if it isn't already
        nucleus_mask = tf.cast(self.mask, dtype=tf.float32)

        # Create a binary mask where pixels with values >= 40 are marked as 1 (cell wall), else 0 (nothing)

        binary_mask_nothing = tf.where(tf.logical_not(tf.logical_and(nucleus_mask>20, nucleus_mask<= 120)), 1, 0)

        binary_mask_nucleus = tf.where(tf.logical_and(nucleus_mask>20, nucleus_mask<= 120), 1,0)

        # One-hot encode the binary mask
        nucleus_mask = tf.concat([binary_mask_nothing, binary_mask_nucleus], axis=2)

        return BinaryMask(nucleus_mask)

    def get_three_classes_mask(self):
        # Convert mask to TensorFlow tensor if it isn't already
        nucleus_mask = tf.cast(self.mask, dtype=tf.float32)

        # Create a binary mask where pixels with values >= 40 are marked as 1 (cell wall), else 0 (nothing)

        binary_mask_nothing = tf.where(nucleus_mask<=20, 1, 0)

        binary_mask_wall = tf.where(tf.logical_and(nucleus_mask>20, nucleus_mask<= 120), 1, 0)

        binary_mask_nucleus = tf.where(nucleus_mask>120, 1,0)

        # One-hot encode the binary mask
        nucleus_mask = tf.concat([binary_mask_nothing, binary_mask_wall,binary_mask_nucleus], axis=2)

        return BinaryMask(nucleus_mask)

    def get_mask_image(self):
        return self.mask
