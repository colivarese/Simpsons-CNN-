from generate_simpsons_dataset import SimpsonsDataset
from DataAugmentation import DataAugmentation
import tensorflow as tf
from CNN import model
from tqdm import tqdm
import numpy as np
import wandb
import datetime
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

wandb.init(project="simpsons-cnn", entity="colivarese")

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#tf_device='/gpu:0'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

simpsons_dataset = SimpsonsDataset()

BATCH_SIZE = 128
SEED = 42
EPOCHS = 1000
NUM_CLASSES = 10
LR = 0.0002 #0.00015

tf.random.set_seed(SEED)


augmentator = DataAugmentation()


train_data, test_data, classes = simpsons_dataset.generate_N_dataset(num_classes=NUM_CLASSES, balance=True)

#simpsons_dataset.check_dataloaders(test_data, classes)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=LR)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(model, tdata, labels):
    with tf.GradientTape() as tape:
        preds = model(tdata)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, preds))

    gradients = tape.gradient(loss, model.trainable_variables)
    capped_grads_and_vars = [(grad,model.trainable_variables[index]) for index, grad in enumerate(gradients)]
    optimizer.apply_gradients(capped_grads_and_vars)

    train_loss(loss)
    #tf.print(preds)
    train_accuracy(labels, preds)

@tf.function
def test_step(model, tdata, labels):
    preds = model(tdata)
    t_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, preds))
    test_loss(t_loss)
    test_accuracy(labels, preds)

# -create log dits
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

def fit(model, train_data,test_data, epochs,seed, batch_size):

    for epoch in tqdm(range(epochs), desc='Training model'):
        train_batch = train_data.shuffle(len(train_data)+1, seed=seed).batch(batch_size)
        train_batch = list(train_batch.as_numpy_iterator())

        for i in range(len(train_batch)):
            epoch_x, epoch_y = train_batch[i][0], train_batch[i][1]
            #epoch_x = np.asarray(augmentator.augment_batch(epoch_x))
            train_step(model,epoch_x,epoch_y)

        test_batch = test_data.batch(216)
        test_batch = list(test_batch.as_numpy_iterator())
        for i in range(len(test_batch)):
            test_step(model, test_batch[i][0], test_batch[i][1])

        template = 'Epoch {}, Perdida: {:.2f}, Exactitud: {:.2f}, Perdida prueba: {:.2f}, Exactitud prueba: {:.2f}'
        print(template.format(epoch+1,
                         train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result()*100, step=epoch)

        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result()*100, step=epoch)

        #wandb.log({'train_accuracy': train_accuracy.result()*100, 'train_loss': train_loss.result(),
          #          'test_accuracy':test_accuracy.result()*100, 'test_loss':test_loss.result()})

        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()


CNN = model(n_classes =NUM_CLASSES)
fit(CNN, train_data=train_data, test_data=test_data, epochs=EPOCHS, seed=SEED, batch_size=BATCH_SIZE)

def predict_correlation_matrix(model, data, classes):

        test_batch = data.batch(216)
        test_batch = list(test_batch.as_numpy_iterator())

        true_labels = list()
        predictions = list()
        
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

predict_correlation_matrix(CNN, test_data, classes)
 