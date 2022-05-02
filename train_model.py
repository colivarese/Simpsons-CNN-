import tensorflow as tf
from tqdm import tqdm


optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)

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
    train_accuracy(labels, preds)

def fit(model, train_data, epochs,seed, batch_size):

    for epoch in tqdm(range(epochs), desc='Training model'):
        i = 0
        train_batch = train_data.shuffle(10, seed=seed).batch(batch_size)
        train_batch = list(train_batch.as_numpy_iterator())

        for i in range(len(train_batch)):
            epoch_x, epoch_y = train_batch[i][0], train_batch[i][1]
            train_step(model,epoch_x,epoch_y)

        template = 'Epoch {}, Perdida: {}, Exactitud: {}'
        print(template.format(epoch+1,
                         train_loss.result(),
                        train_accuracy.result()*100))

        train_loss.reset_states()
        train_accuracy.reset_states()