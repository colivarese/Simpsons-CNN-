from generate_simpsons_dataset import SimpsonsDataset #-Se crea una clase para generar los Datasets de entrenamiento
from DataAugmentation import DataAugmentation #-Se crea una clase para generar algunos cambios en las imágenes
import tensorflow as tf # -Se importa Tensorflow, la herramienta a utilizar
from CNN import model # -Se crea la arquitectura a utilizar en una clase 
from tqdm import tqdm #- Se importa tqdm para tener una mejor visualización del tiempo de entrenamiento
import numpy as np #-Se importa numpy para el manejo númerico
import wandb # -se importa wandb como un segundo método de monitoreo del entrenamiento
import datetime # -se importa datetime para dar formato de tiempo a los logs de TensorBoard
import matplotlib.pyplot as plt # -Se importa pyplot para la visualización de la matriz de confusión
import seaborn as sns # -se importa seaborn para la visualización de la matriz de confusión
from sklearn.metrics import confusion_matrix # -se importa para generar la matriz de confusión

wandb.init(project="simpsons-cnn", entity="colivarese") #-se inicializa una corrida en el projecto Simpsons-cnn de wandb


config = tf.compat.v1.ConfigProto() # -Se inicializa un objeto para la configuración del uso de TensorFlow
config.gpu_options.allow_growth=True # -se permite el aumento de la memoria de la GPU
sess = tf.compat.v1.Session(config=config) # -se asigna la configuración propuesta a la GPU

simpsons_dataset = SimpsonsDataset() # -Se crea el objeto para generar los datasets

BATCH_SIZE = 128    # -Se define un batch size (Se encontro de forma empírica encontrando un mejor desempeño y velocidad con 128)
SEED = 42  # -se define una semilla para tener reproducibilidad
EPOCHS = 900 # -se define el número de épocas para entrenar
NUM_CLASSES = 10 # -se define el número de clases a utilizar
LR = 0.0002 #0.00015 # -se define una tasa de aprendizaje, esta se encontro de forma empirica.

tf.random.set_seed(SEED) # -se asigna la semilla aleatoria


augmentator = DataAugmentation() #- Se crea el objeto de Data Agumentation


train_data, test_data, classes = simpsons_dataset.generate_N_dataset(num_classes=NUM_CLASSES, balance=False)
 # - se generan los datos de entrenamiento y de prueba, esta función toma como parametros el número de clases a tomar del conjunto de datos total
 # y un parametro de balance, el cual si es True tomara 1000 imágenes unicamente por clase y si es False, tomara todas las imágenes disponibles,
 # esto se implemento para intentar solucionar el problema del desbalance de las clases, donde unas tenian muchas más imágenes que otras.
 # - La función regresa dos objetos de Dataset de Tensorflow, de entrenamiento y prueba y las clases con las que se esta trabajando.

#simpsons_dataset.check_dataloaders(test_data, classes)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=LR) # -se inicializa el optimizador a utilizar, en este caso un Adam y se asigna la tasa de aprendizaje.

train_loss = tf.keras.metrics.Mean(name='train_loss') # -se define la función de perdida del entrenamiento
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy') #-se define la función de exactitud de entrenamiento
test_loss = tf.keras.metrics.Mean(name='test_loss') # -se define la función de perdida de la prueba
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy') # -se define la función de exactitud de la prueba

@tf.function
def train_step(model, tdata, labels): #-toma como parametros el modelo a entrar, los datos de entrenamiento y sus etiquetas verdaderas.
    with tf.GradientTape() as tape: # -se inicializa la cinta para registrar los gradientes
        preds = model(tdata) # -se realizan predicciones con el modelo y los datos de entrenamiento

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, preds)) # -se calcula función de pérdida utilizando las predicciones y las etiquetas verdaderas.

    gradients = tape.gradient(loss, model.trainable_variables) # -se calculan los gradientes con respecto a cada variable entrenable
    capped_grads_and_vars = [(grad,model.trainable_variables[index]) for index, grad in enumerate(gradients)] #- Se ordena cada gradiente con su variable
    optimizer.apply_gradients(capped_grads_and_vars) # -se aplica un paso del optimizador con los gradientes.

    train_loss(loss) # -se registra la perdida del paso de entrenamiento
    train_accuracy(labels, preds) # -se registra la exactitud del paso de entrenamiento

@tf.function
def test_step(model, tdata, labels): #-toma como parametros el modelo a entrar, los datos de entrenamiento y sus etiquetas verdaderas
    preds = model(tdata) # -se realizan predicciones con el modelo y los datos de entrenamiento
    t_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, preds)) # -se calcula función de pérdida utilizando las predicciones y las etiquetas verdaderas.
    test_loss(t_loss)  # -se registra la perdida del paso de prueba
    test_accuracy(labels, preds)  # -se registra la exactitud del paso de prueba

# -create log dits
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # -se registra el tiempo actual
train_log_dir = 'logs/gradient_tape/' + current_time + '/train' #- se crea un folder para guardar los registros de entrenamiento
test_log_dir = 'logs/gradient_tape/' + current_time + '/test' # -se crea un folder para guardar los registros de prueba
train_summary_writer = tf.summary.create_file_writer(train_log_dir) # -se escriben los registros de entrenamiento
test_summary_writer = tf.summary.create_file_writer(test_log_dir) # -se escriben los registros de prueba

# -se define la función para entrenenar el modelo, esta toma como parametros el modelo a entrar, los Dataset previamente creados 
# de entrenamiento y prueba, el número de épocas a entrenar, una semilla aleatoria y el tamaño del batch.
def fit(model, train_data,test_data, epochs,seed, batch_size):

    for epoch in tqdm(range(epochs), desc='Training model'): # -por el número de epocas definidio
        train_batch = train_data.shuffle(len(train_data)+1, seed=seed).batch(batch_size) # -se revuelven los datos de entrenamiento y se toma un batch
        train_batch = list(train_batch.as_numpy_iterator()) # -se transforma el batch a una lista como iterador de numpy

        for i in range(len(train_batch)): # -iterando sobre la lista de datos de entrenamiento
            epoch_x, epoch_y = train_batch[i][0], train_batch[i][1] # -se guardan los datos de entrenamiento en epoch_x y las etiquetas reales en epoch_y
            #epoch_x = np.asarray(augmentator.augment_batch(epoch_x)) # -Esta linea se puede descomentar para utilizar el DataAgumentation, pero el desempeño es peor.
            train_step(model,epoch_x,epoch_y) # -Se hace un paso de entrenamiento
        
        # - ...una vez que se entreno en todo el batch de entrenamiento
        test_batch = test_data.batch(216) # - se toman batches para los datos de prueba
        test_batch = list(test_batch.as_numpy_iterator()) # - se transforma una lista de un iterador de numpy
        for i in range(len(test_batch)): # -iterando sobre la lista de datos de prueba
            test_step(model, test_batch[i][0], test_batch[i][1]) # -se realizan pasos de prueba

        template = 'Epoch {}, Perdida: {:.2f}, Exactitud: {:.2f}, Perdida prueba: {:.2f}, Exactitud prueba: {:.2f}' # -se define una string para imprimir el desempeño
        print(template.format(epoch+1,
                         train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))

        with train_summary_writer.as_default(): # -se registra el desempeño de entrenamiento para el TensorBoard
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result()*100, step=epoch)

        with test_summary_writer.as_default(): #- se registra el desempeño de prueba para el TensorBoard
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result()*100, step=epoch)

        wandb.log({'train_accuracy': train_accuracy.result()*100, 'train_loss': train_loss.result(), #- Para registrar las pruebas en wandb
                    'test_accuracy':test_accuracy.result()*100, 'test_loss':test_loss.result()})

        # -se resetean los objetos de perdida y exactitud tanto de prueba y entrenamiento.
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()


CNN = model(n_classes =NUM_CLASSES) # -se crea el modelo de red convolucional, tomando como parametro el número de clases
fit(CNN, train_data=train_data, test_data=test_data, epochs=EPOCHS, seed=SEED, batch_size=BATCH_SIZE) # -Se entrena la red

# Se define una función para generar la matriz de confusión y poder tener una mejor visualización del desempeño de la red.
def predict_correlation_matrix(model, data, classes):

        test_batch = data.batch(216)
        test_batch = list(test_batch.as_numpy_iterator())

        true_labels = list()
        predictions = list()
        
        # -En general es similar a la de entrenamiento, pero en este caso se hace una sola corrida de todos los datos y se almacenan tanto las 
        # predicciones como las etiquetas reales, para posteriormente generar un mapa de correlación entre todas las variables.
        for i in range(len(test_batch)):
            labels = test_batch[i][1]
            preds = model(test_batch[i][0]).numpy()
            preds = tf.nn.softmax(preds).numpy()
            for label, pred in zip(labels, preds):
                true_labels.append(np.argmax(label))
                predictions.append(np.argmax(pred))
        
        classes = list(classes.keys())
        ax= plt.subplot()
        cf = confusion_matrix(true_labels, predictions)
        sns.heatmap(cf, annot=True, fmt='g', ax=ax, cbar=False, cmap='PiYG')
        
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix') 
        ax.xaxis.set_ticklabels(classes)
        ax.yaxis.set_ticklabels(classes)
        plt.xticks(rotation = 45)
        plt.yticks(rotation = 45)

        plt.show()

# -se hace la matriz de confusión con el modelo entrenado y los datos de prueba.
predict_correlation_matrix(CNN, test_data, classes)
 