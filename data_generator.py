import os
import numpy as np
import random
from glob import glob
import collections
import tensorflow as tf
from tqdm import tqdm
import albumentations as A

class SegmentationDataGenerator():
    def __init__(self,
                image_directory,
                segmentation_directory,
                num_classes,
                batch_size,
                augmentation = False,
                normalize = True,
                shuffle = True
                        ):
        self.image_directory = image_directory
        self.segmentation_directory = segmentation_directory
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.normalize = normalize
        self.shuffle = shuffle

        print(f"Indexing Image files...")
        self.img_files = sorted(glob(f"{image_directory}"))
        
        print(f"Indexing Segmentation files...")
        self.segmentation_files = sorted(glob(f"{segmentation_directory}"))

        print(f"Loaded {len(self.img_files)} images with {len(self.segmentation_files)} segmentations")
        
        if len(self.img_files) != len(self.segmentation_files):
            raise Exception(f"Number of images ({len(self.img_files)}) is not equal to number of segmentations ({len(self.segmentation_files)}). ")
        
        assert len(self.img_files) >= self.batch_size; "Batch Size must be smaller than number of images"

        self.data = list(zip(self.img_files, self.segmentation_files))
        del self.img_files
        del self.segmentation_files

    def __len__(self) -> int:
        return len(self.data)
    
    def normalizer(self, image):
        image =  (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-9)
        #image = (image - np.mean(image) * np.max(image)) / (np.std(image) * np.max(image))
        return image
                
    def get_data_generator(self):

        L = self.__len__()

        #keras needs the generator infinite, so we will use while true  
        while True:
            
            # shuffle
            if self.shuffle == True:
                random.shuffle(self.data)

            batch_start = 0
            batch_end = self.batch_size

            while batch_start + self.batch_size <= L:
                limit = min(batch_end, L)
                
                images = []
                segmentations = []
                
                for image_path, segmentation_path in self.data[batch_start:limit]:
                    # load image
                    i = tf.keras.utils.load_img(image_path)
                    image = self.normalizer(np.array(i).astype('float32'))
                    
                    # load segmentation
                    i = tf.keras.utils.load_img(segmentation_path, color_mode = "grayscale")
                    segmentation = np.array(i).astype('float32')

                    # perform augmentation
                    
                    if self.augmentation != False:

                      augmented = self.augmentation(image=np.array(image), mask=np.array(segmentation))
                    
                      image = augmented['image']
                      segmentation = augmented['mask']

                    #normalize the image
                    image = self.normalizer(image) 
                    
                    images.append(image)
                    segmentations.append(segmentation) 
                
                images = np.asarray(images)
                segmentations = np.asarray(segmentations)

                # to categorical (used for maskes) - basically one hot encoding
                categories = tf.keras.utils.to_categorical(segmentations, num_classes=self.num_classes)
                segmentations = categories.reshape((segmentations.shape[0], segmentations.shape[1], segmentations.shape[2], self.num_classes))


                yield (images, segmentations) 

                batch_start += self.batch_size   
                batch_end += self.batch_size