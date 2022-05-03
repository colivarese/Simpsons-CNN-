from generate_simpsons_dataset import SimpsonsDataset
from DataAugmentation import DataAugmentation
import train_model
import tensorflow as tf
from CNN import model
from tqdm import tqdm
import numpy as np
import wandb

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
EPOCHS = 350
NUM_CLASSES = 10
LR = 0.0002 #0.00015

tf.random.set_seed(SEED)


augmentator = DataAugmentation()


train_data, test_data = simpsons_dataset.generate_N_dataset(num_classes=NUM_CLASSES, balance=True)

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

def fit(model, train_data,test_data, epochs,seed, batch_size):

    for epoch in tqdm(range(epochs), desc='Training model'):
        train_batch = train_data.shuffle(len(train_data)+1, seed=seed).batch(batch_size)
        train_batch = list(train_batch.as_numpy_iterator())

        for i in range(len(train_batch)):
            epoch_x, epoch_y = train_batch[i][0], train_batch[i][1]
            epoch_x = np.asarray(augmentator.augment_batch(epoch_x))
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

        wandb.log({'train_accuracy': train_accuracy, 'train_loss': train_loss,
                    'test_accuracy':test_accuracy, 'test_loss':test_loss})

        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()


CNN = model(n_classes =NUM_CLASSES)
fit(CNN, train_data=train_data, test_data=test_data, epochs=EPOCHS, seed=SEED, batch_size=BATCH_SIZE)
