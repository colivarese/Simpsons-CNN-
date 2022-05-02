from cgi import test
import os
import pandas as pd
import numpy as np
import PIL
from PIL import Image
from tqdm import tqdm
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class SimpsonsDataset(object):
    def __init__(self):
        self.dataset_path = 'simpsons_dataset/simpsons_dataset'

        classes = os.listdir(self.dataset_path)

        image_per_class = []
        for cl in classes:
            class_path = f'{self.dataset_path}/{cl}'
            num_images = len(os.listdir(class_path))
            image_per_class.append(num_images)

        df = pd.DataFrame(list(zip(classes, image_per_class)), columns=['class','num_images'])
        df = df.sort_values(by=['num_images'], ascending=False)

        self.dataset = df


    def generate_N_dataset(self, num_classes, balance=False):

        total_balance = 1000

        if balance:
            self.dataset = self.dataset.assign(num_images = total_balance)

        classes = list(self.dataset.head(num_classes)['class'].values)

        total_num_images = self.dataset.head(num_classes)['num_images'].sum()

        n_w, n_h = 64, 64
        X = np.zeros((total_num_images, n_w, n_h, 3))
        y = np.zeros(total_num_images)
        num_image = 0

        idx_class = np.arange(0,num_classes)

        classes_with_idx = dict(zip(classes,idx_class))

        for cl in tqdm(classes, desc='Generating training set'):
            class_path = f'{self.dataset_path}/{cl}'
            if balance:
                class_images = os.listdir(class_path)[:total_balance]
            else:
                class_images = os.listdir(class_path)
            for img in class_images:
                image = (Image.open(f'{class_path}/{img}')).convert('RGB')
                if image.mode == 'RGB':
                    image = image.resize((n_w,n_h), Image.BILINEAR)
                    image = np.array(image, dtype=np.uint8)
                    X[num_image] = image
                    idx_class = classes_with_idx[cl]
                    y[num_image] = classes_with_idx[cl]
                    num_image += 1


        X = X / 255
        y = y.astype('uint8')
        y_hot = np.eye(num_classes)
        y_hot = y_hot[y]

        test_data = self.generate_test_set(classes=classes, classes_with_idx=classes_with_idx)

        train_data = tf.data.Dataset.from_tensor_slices((X,y_hot))

        return train_data, test_data

    def generate_test_set(self, classes, classes_with_idx):
        test_path = 'simpsons_testdataset/simpsons_testdataset'

        total_num_images = 0
        for cl in classes:
            class_path = f'{test_path}/{cl}'
            num_images = len(os.listdir(class_path))
            total_num_images += num_images

        n_w, n_h = 64, 64
        X = np.zeros((total_num_images, n_w, n_h, 3))
        y = np.zeros(total_num_images)
        num_image = 0

        for cl in tqdm(classes, desc='Generating test set'):
            class_path = f'{test_path}/{cl}'
            class_images = os.listdir(class_path)
            for img in class_images:
                image = (Image.open(f'{class_path}/{img}')).convert('RGB')
                if image.mode == 'RGB':
                    image = image.resize((n_w,n_h), Image.BILINEAR)
                    image = np.array(image, dtype=np.uint8)
                    X[num_image] = image
                    y[num_image] = classes_with_idx[cl]
                    num_image += 1

        num_classes = len(classes)
        X = X / 255
        y = y.astype('uint8')
        y_hot = np.eye(num_classes)
        y_hot = y_hot[y]

        test_data = tf.data.Dataset.from_tensor_slices((X,y_hot))

        return test_data


        

        

    

