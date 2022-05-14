import tensorflow as tf
import random
import os
import numpy

class DataAugmentation(object):
    def __init__(self):
        self.brightness_delta = 0.2
        self.low_contrast = 0.2
        self.up_contrast = 0.5
        self.r_crop = (1,3)
        self.low_sat = 0.2
        self.up_sat = 0.5

    def augment_batch(self, batch):
        """
        Esta función toma como entrada un batch de datos y les realiza una transformación dependiendo
        de una probabilidad de 0.5
        Las transformaciones que se realizan son cambios de brillo, contraste, recortes aleatorios y saturación.

        Parametros:
        batch: El conjunto de datos a transformar.
        """
        p = [random.randint(0, 1) for _ in range(5)]
        if p[0] == 1:
            fn = lambda x: tf.image.random_brightness(x, self.brightness_delta)
            batch = tf.map_fn(fn, batch)
        if p[1] == 1:
            fn = lambda x: tf.image.random_contrast(x, self.low_contrast, self.up_contrast)
            batch = tf.map_fn(fn, batch)
        #elif p[2] == 1:
         #   fn = lambda x: tf.image.random_crop(x, self.r_crop)
          #  batch = tf.map_fn(fn, batch)
        if p[2] == 1:
            fn = lambda x: tf.image.random_saturation(x, self.low_sat, self.up_sat)
            batch = tf.map_fn(fn, batch)
        if p[3] == 1:
            batch = tf.image.random_flip_up_down(batch)
        if p[4] == 1:
            batch = tf.image.random_flip_left_right(batch)

        return batch


    

