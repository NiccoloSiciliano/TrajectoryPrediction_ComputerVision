import tensorflow as tf
from tensorflow.keras import layers, models
from CellMask import CellMask
import os
from sklearn.cluster import KMeans
from var import n_classes, clas, activation
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.config.run_functions_eagerly(True)


class Utils:

    def one_hot_vec(mask):
        h,w,_ = mask.shape

        one_hot_vec = []
        for i in range(h):
            one_hot_vec.append([])
            for j in range(w):
                v = [0,0,0]
                v[np.argmax(mask[i][j])] = 1
                one_hot_vec[i].append(v)
        return np.array(one_hot_vec)

    def create_classification_dataset(images_dir):
        dataset_datagen = ImageDataGenerator(rescale=1./255)
        dataset = dataset_datagen.flow_from_directory(
            images_dir,  # Path to training data
            target_size=(256, 256),  # Resize images
            batch_size=32,
            class_mode='categorical'  # Since there are multiple classes
        )
        return dataset

    def nucleus_pred(image, mask):
        mask_wall = mask.get_cell_wall_mask()
        
        image = mask_wall.get_masked_image(image)/255.0
      
        return image,mask.get_nucleus_mask().get_binary_mask()

    def wall_pred(image, mask):
        image =  tf.cast(image, tf.float32)/255.0
        return image, mask.get_cell_wall_mask().get_binary_mask()
    
    def three_classes(image, mask):
        image =  tf.cast(image, tf.float32)/255.0
        return image, mask.get_three_classes_mask().get_binary_mask()
    
    def load_image(image_path, mask_path):
        w = h = 256
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [256, 256])
        image = tf.cast(image, tf.uint8)
    
        # Load mask
        mask = CellMask(mask_path)
        pre_proc = Utils.three_classes if n_classes == 3 else Utils.wall_pred if clas == "WALL" else Utils.nucleus_pred
        print(pre_proc)

        return pre_proc(image, mask)

    def load_classification_image(image_path):
        w = h = 256
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [256, 256])
        image =  tf.cast(image, tf.float32)/255.0

        return image
    
    def accuracy(groundtruth_mask, pred_mask):
        h,w,_ = groundtruth_mask.shape
        tot = 0
        matches= 0
        wall_tot = 0
        wall_matches = 0
        nuc_tot = 0
        nuc_matches = 0

        for i in range(h):
            for j in range(w):

                if np.argmax(groundtruth_mask[i][j]) != 0:
                    if np.argmax(groundtruth_mask[i][j]) == 2:
                        wall_tot+=1
                    else:
                        nuc_tot+=1
                    tot+=1

                if np.argmax(groundtruth_mask[i][j]) == np.argmax(pred_mask[i][j]) != 0:
                    if np.argmax(groundtruth_mask[i][j]) == 2:
                        wall_matches+=1
                    else:
                        nuc_matches+=1
                    matches+=1


        return matches/tot

    def dice_coefficient(y_true, y_pred, smooth=1e-6):
        """
        Calculate the Dice Coefficient.

        Args:
            y_true (tensor): Ground truth mask (batch size, height, width, num_classes).
            y_pred (tensor): Predicted mask (batch size, height, width, num_classes).
            smooth (float): Smoothing factor to avoid division by zero.

        Returns:
            dice (tensor): Dice coefficient.
        """
        # Ensure both y_true and y_pred are of type float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Flatten the predictions and ground truth masks
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])

        # Calculate the intersection and union
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)

        # Calculate Dice Coefficient
        dice = (2. * intersection + smooth) / (union + smooth)

        return dice
    def multi_class_dice_coefficient(y_true, y_pred, smooth=1e-6):
        """
        Calculate the Dice coefficient for multi-class segmentation.

        Parameters:
        - y_true: Ground truth one-hot encoded labels, shape (batch_size, height, width, num_classes).
        - y_pred: Predicted one-hot encoded labels or softmax probabilities, shape (batch_size, height, width, num_classes).
        - smooth: Smoothing factor to avoid division by zero.

        Returns:
        - Dice coefficient for each class as a list.
        """

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Calculate Dice coefficient for each class
        dice_per_class = []
        for i in range(y_true.shape[-1]):
            y_true_f = y_true[..., i]
            y_pred_f = y_pred[..., i]

            intersection = tf.reduce_sum(y_true_f * y_pred_f)
            union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

            dice = (2. * intersection + smooth) / (union + smooth)
            dice_per_class.append(dice)
        if y_true.shape[-1] == 2:
            return dice_per_class[0]*0.3+ dice_per_class[1]*0.7
        return dice_per_class[0]*0.1+ dice_per_class[1]*0.35 + dice_per_class[2]*0.55

    def create_dataset(images_dir, masks_dir):
        image_paths = sorted([os.path.join(images_dir, fname) for fname in os.listdir(images_dir)])
        mask_paths = sorted([os.path.join(masks_dir, fname) for fname in os.listdir(masks_dir)])
        print(len(image_paths), len(mask_paths))
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
        dataset = dataset.map(lambda image_path, mask_path: Utils.load_image(image_path, mask_path))

        return dataset
    
    def get_arch3(n_classes):
        inputs = layers.Input(shape=(256, 256, 3))
        # Encoder
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        skip1 = x
        
        #net1
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        skip2 = x
        
        #net2
        x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        skip3 = x
        
        #Bottleneck
        x = layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x_o = layers.Concatenate()([skip3, x])
        
        #NUCLEUS net2 (dec)
        x_n = layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x_o)
        x_n = layers.BatchNormalization()(x_n)
        x_n = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x_n)
        
        #WALL net2 (dec)
        x_w = layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x_o)
        x_w = layers.BatchNormalization()(x_w)
        x_w = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x_w)
        
        #NUCLEUS net1 (dec)
        x_n = layers.Concatenate()([ skip2,x_n])
        x_n = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x_n)
        x_n = layers.BatchNormalization()(x_n)
        x_n = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x_n)
        x_n = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x_n)
        #WALL net1 (dec)
        x_w = layers.Concatenate()([skip2, x_w])
        x_w = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x_w)
        x_w = layers.BatchNormalization()(x_w)
        x_w = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x_w)
        x_w = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x_w)
        
        
        
        x = layers.Concatenate()([skip1, x_n, x_w])
        x = layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)


        # Output Layer for Segmentation
        outputs = layers.Conv2D(filters=n_classes, kernel_size=(1, 1), activation=activation, padding='same')(x)

        return models.Model(inputs=inputs, outputs=outputs)
    
    def get_arch1(n_classes):
        inputs = layers.Input(shape=(256, 256, 3))

        # Encoder
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        skip1 = x

        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        skip2 = x

        x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        skip3 = x

        # Bottleneck
        x = layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)

        # Decoder
        x = layers.Concatenate()([skip3, x])
        x = layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

        
        x = layers.Concatenate()([skip2, x])
        x = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    
        x = layers.Concatenate()([skip1, x])
        x = layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)


        # Output Layer for Segmentation
        outputs = layers.Conv2D(filters=n_classes, kernel_size=(1, 1), activation=activation, padding='same')(x)

        return models.Model(inputs=inputs, outputs=outputs)
    
    def get_wall_arch(n_classes):
        inputs = layers.Input(shape=(256, 256, 3))

        # Encoder
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        skip1 = x

        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        skip2 = x

        x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        skip3 = x

        # Bottleneck
        x = layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
        

        # Decoder
        x = layers.Concatenate()([skip3, x])
        x = layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        

        
        x = layers.Concatenate()([skip2, x])
        x = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        
    
        x = layers.Concatenate()([skip1, x])
        x = layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)


        # Output Layer for Segmentation
        outputs = layers.Conv2D(filters=n_classes, kernel_size=(1, 1), activation=activation, padding='same')(x)

        return models.Model(inputs=inputs, outputs=outputs)
    def get_nucleus_arch(n_classes):
        inputs = layers.Input(shape=(256, 256, 3))

        # Encoder
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        skip1 = x

        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        skip2 = x
        
        # Bottleneck
        
        x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        
        
        # Decoder
        x = layers.Concatenate()([skip2, x])
        x = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        
        x = layers.Concatenate()([skip1, x])
        x = layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)


        # Output Layer for Segmentation
        outputs = layers.Conv2D(filters=n_classes, kernel_size=(1, 1), activation=activation, padding='same')(x)

        return models.Model(inputs=inputs, outputs=outputs)

    def get_arch_classification(n_classes):
        inputs = layers.Input(shape=(256, 256, 3))

        # First Block
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(pool_size=(2,2))(x)

        # Second Block
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(pool_size=(2,2))(x)

        # Third Block
        x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(pool_size=(2,2))(x)

        # Fully Connected Layers
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.15)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.05)(x)

        # Output Layer
        outputs = layers.Dense(5, activation='softmax')(x)
        
        return models.Model(inputs=inputs, outputs=outputs)