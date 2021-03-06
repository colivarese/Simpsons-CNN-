import tensorflow as tf
import numpy as np



class model(object):
    """
    En esta clase se define la arquitectura de la red neuronal convolucional.
    Esta consta de 6 capas ocultas y una capa totalmente conectada.
    """
    def __init__(self, n_classes = 10, H = 64):

        self.fhl1 = 32 # -Primera capa oculta
        self.fhl2 = 32 #- Segunda capa oculta
        self.fhl3 = 64 # -Tercera capa oculta
        self.fhl4 = 64 #-Cuarta capa oculta
        self.fhl5 = 256 # -Quinta capa oculta
        self.fhl6 = 64 # -sexta capa oculta

        self.ffc1 = 512 # -Primera capa totalmente conectada

        self.last = 64 # -Ultima capa totalmente conectada

        self.n_w = H # -nuevas dimensiones de la imagen (ancho)
        self.n_h = H # nuevas dimensiones de la imagen (alto)

        self.hl1W = tf.Variable(np.random.rand(3,3,3,self.fhl1), name='hl1w', dtype='float32')
        self.hl1B = tf.Variable(np.random.rand(self.fhl1), name='hl1b', dtype='float32')

        self.hl2W = tf.Variable(np.random.rand(3,3,self.fhl1,self.fhl2), name='hl2w', dtype='float32')
        self.hl2B = tf.Variable(np.random.rand(self.fhl2), name='hl2b', dtype='float32')

        self.hl3W = tf.Variable(np.random.rand(3,3,self.fhl2,self.fhl3), name='hl3w', dtype='float32')
        self.hl3B = tf.Variable(np.random.rand(self.fhl3), name='hl3b', dtype='float32')

        self.hl4W = tf.Variable(np.random.rand(3,3,self.fhl3,self.fhl4), name='hl4w', dtype='float32')
        self.hl4B = tf.Variable(np.random.rand(self.fhl4), name='hl4b', dtype='float32')

        self.hl5W = tf.Variable(np.random.rand(3,3,self.fhl4,self.fhl5), name='hl5w', dtype='float32')
        self.hl5B = tf.Variable(np.random.rand(self.fhl5), name='hl5b', dtype='float32')

        self.hl6W = tf.Variable(np.random.rand(3,3,self.fhl5,self.fhl6), name='hl6w', dtype='float32')
        self.hl6B = tf.Variable(np.random.rand(self.fhl6), name='hl6b', dtype='float32')

        self.fc1W = tf.Variable(np.random.rand(27*27*self.fhl6,self.ffc1), name='fc1W', dtype='float32')
        self.fc1B = tf.Variable(np.random.rand(self.ffc1), name='fc1B', dtype='float32')

        self.outW = tf.Variable(np.random.rand(self.ffc1, n_classes), name='outW', dtype='float32')
        self.outB = tf.Variable(np.random.rand(n_classes), name='outB', dtype='float32')

        # -se definen las variables entrenables
        self.trainable_variables = [self.hl1W, self.hl1B,
                                   self.hl2W, self.hl2B,
                                   self.hl3W, self.hl3B,
                                   self.hl4W, self.hl4B,
                                   self.hl5W, self.hl5B,
                                   self.hl6W, self.hl6B,
                                   self.fc1W, self.fc1B,
                                   self.outW, self.outB]
        
    def __call__(self,x):
        
        with tf.device('/device:GPU:0'):

            """
            Esta secci??n es importante de considerar, ya que es la forma en que esta trabajando la red,
            los tensores pasaran por cada capa.
            Se notaba al principio que la red tenia un overfitting muy pronto, por lo que se considero que quizas el n??mero 
            de parametros era muy alto y que pod??a facilmente esta aprendiendo de la red, se movio esto, pero no se tuvo mejora, 
            por lo que se considero que se estaba aprendiendo las im??genes y no encontraba cada vez rasgos de m??s alto nivel, 
            por lo que se realizo un cambio en el tama??o de los max pooling para tener una mayor agregaci??n de informaci??n y as??
            enfocarse en rasgos de alto nivel. Esto ayudo a que se mejorara el desempe??o.

            El punto m??s importante fue la utilizaci??n de la normalizaci??n del batch, en donde se normalizan los valores de los mapas
            de activaci??n, esto con el fin de tener una distribuci??n m??s controlada lo que permite a la red tener un mayor enfoque
            en los patrones encontrados y no importantes por su escala, asi mismo una distribuci??n constante permite que el entrenamiento
            sea m??s veloz. Antes de utilizar esta t??cnica la exactitud no aumentaba del 30%, utilizandolo se alcanzan los valores de alrededor de 80%.

            Se definio un valor de dropout chico, uno m??s alto no permitia a la red aprender, o quiz??s si pero en m??s ??pocas.
            """

            dropout = 0.1
        
            x = tf.cast(x, tf.float32)
            img = tf.reshape(x, shape=[-1,self.n_w,self.n_h,3])
                        
            l1 = tf.nn.conv2d(img, self.hl1W, strides= [1,1,1,1], padding='SAME')
            l1 = tf.add(l1, self.hl1B)
            l1 = tf.nn.relu(l1)

            
            l2 = tf.nn.conv2d(l1,self.hl2W, strides=[1,1,1,1], padding='SAME')
            l2 = tf.add(l2, self.hl2B)
            l2 = tf.nn.relu(l2)


            l2 = tf.nn.max_pool2d(l2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
            mean_x, std_x = tf.nn.moments(l2, axes = 2, keepdims=True)
            l2 = tf.nn.batch_normalization(l2, mean_x, std_x, None, None, 1e-12)

            l2 = tf.nn.dropout(l2, dropout) #-dropout

            
            l3 = tf.nn.conv2d(l2,self.hl3W, strides=[1,1,1,1], padding='SAME')
            l3 = tf.add(l3, self.hl3B)
            l3 = tf.nn.relu(l3)
            
            
            l4 = tf.nn.conv2d(l3,self.hl4W, strides=[1,1,1,1], padding='VALID')
            l4 = tf.add(l4, self.hl4B)
            l4 = tf.nn.relu(l4)


            l4 = tf.nn.max_pool2d(l4, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
            mean_x, std_x = tf.nn.moments(l4, axes = 2, keepdims=True)
            l4 = tf.nn.batch_normalization(l4, mean_x, std_x, None, None, 1e-12)

            l4 = tf.nn.dropout(l4, dropout) # -dropout
            
            
            l5 = tf.nn.conv2d(l4,self.hl5W, strides=[1,1,1,1], padding='SAME')
            l5 = tf.add(l5, self.hl5B)
            l5 = tf.nn.relu(l5)

            

            l6 = tf.nn.conv2d(l5, self.hl6W, strides=[1,1,1,1], padding='VALID')
            l6 = tf.add(l6, self.hl6B)
            l6 = tf.nn.relu(l6)


            l6 = tf.nn.max_pool2d(l6, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
            mean_x, std_x = tf.nn.moments(l6, axes = 2, keepdims=True)
            l6 = tf.nn.batch_normalization(l6, mean_x, std_x, None, None, 1e-12)

            
            l7 = tf.reshape(l6, [-1, 27*27*self.fhl6])
            l7 = tf.matmul(l7, self.fc1W)
            l7 = tf.add(l7, self.fc1B)
            l7 = tf.nn.relu(l7)

            l7 = tf.nn.dropout(l7, dropout)  # -dropout

            print(l7.shape)
            out = tf.matmul(l7, self.outW) + self.outB
            return out

    
