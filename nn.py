import miniflow as mf
import numpy as np
import keras

np.random.seed(0)

num_classes = 10
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# normalize data
x_train = x_train.astype(np.float32).reshape(-1, 28*28) / 255
x_test = x_test.astype(np.float32).reshape(-1, 28*28) / 255

# convert to one_hot encoding
y_train_hot = np.zeros(y_train.shape + (num_classes,))
for idx, y in enumerate(y_train):
    y_train_hot[idx, y] = 1

y_test_hot = np.zeros(y_test.shape + (num_classes,))
for idx, y in enumerate(y_test):
    y_test_hot[idx, y] = 1

X, Y = mf.Input(), mf.Input()
W, b = mf.Input(), mf.Input()

linear = mf.Linear(X, W, b)
# activation = mf.Softmax(linear)
cost = mf.CategoricalCrossentropyWithSoftmax(linear, Y)

def get_batches(x, y, batch_size):
    num_samples = len(x)
    indices = np.arange(num_samples)
    while True:
        np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            end = min(num_samples, start + batch_size)
            keys = indices[start:end]
            yield x[keys], y[keys]

epochs = 100
batch_size = 100
steps_per_epoch = int(np.ceil(len(x_train) / float(batch_size)))
W_ = np.random.randn(28*28, num_classes)
b_ = np.random.randn(num_classes)

gen = get_batches(x_train, y_train_hot, batch_size)

trainables = [W, b]
for epoch in range(epochs):
    loss = 0
    correct = 0.
    for step in range(steps_per_epoch):
        x,y = next(gen)
        feed_dict = {
            X: x,
            Y: y,
            W: W_,
            b: b_,
        }
        graph = mf.topological_sort(feed_dict)
        mf.forward_and_backward(graph)
        mf.sgd_update(trainables, learning_rate=1e-4)
        loss += graph[-1].value
        pred = np.argmax(linear.value, axis=-1)
        true = np.argmax(y, axis=-1)
        correct += np.sum(pred == true)

    print("Epoch: {}, Loss: {:.3f}".format(epoch + 1, loss / steps_per_epoch))
    print "Accuracy:", correct / len(x_train) * 100.

    # calculate validation loss and accuracy
    feed_dict = {
        X: x_test,
        Y: y_test_hot,
        W: W_,
        b: b_,
    }
    graph = mf.topological_sort(feed_dict)
    mf.forward_and_backward(graph)
    pred = np.argmax(linear.value, axis=-1)
    true = np.argmax(y_test_hot, axis=-1)
    print 'validation loss:', graph[-1].value
    print 'validation accuracy:', float(np.sum(pred==true)) / len(x_test) * 100
    print

pass