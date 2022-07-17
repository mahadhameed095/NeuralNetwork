import numpy as np
from Network import Network
from Trainer import Trainer
from BinaryCrossEntropy import BinaryCrossEntropy
import matplotlib.pyplot as plt
# import time
from Utils import oneHotEncode


def MNIST_readImages(path: str, numToRead: int) -> np.ndarray:
    with open(path, "rb") as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        assert magic_number == 2051, "Incorrect format file."
        num_images = int.from_bytes(f.read(4), 'big')
        assert numToRead <= num_images, "Maximum number of images exceeded"
        f.read(8)  # skipping 2 integers
        data = np.frombuffer(f.read(784 * numToRead), dtype=np.uint8).astype(np.float32)
        return data.reshape(numToRead, 784)


def MNIST_readLabels(path: str, numToRead: int) -> np.ndarray:
    with open(path, "rb") as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        assert magic_number == 2049, "Incorrect format file."
        num_labels = int.from_bytes(f.read(4), 'big')
        assert numToRead <= num_labels, "Maximum number of labels exceeded"
        return np.frombuffer(f.read(numToRead), dtype=np.uint8).reshape(numToRead, 1)

def calcAccuracy(forwardMat: np.ndarray, train_Y: np.ndarray):
    preds = np.argmax(forwardMat, 1)
    num_correct_pred = preds == np.argmax(train_Y, 1)
    num_correct_pred = np.count_nonzero(num_correct_pred)
    return (num_correct_pred) * 100 / train_Y.shape[0]


paths = [
    "Datasets/MNIST/train-images-idx3-ubyte",
    "Datasets/MNIST/train-labels-idx1-ubyte",
    "Datasets/MNIST/t10k-images-idx3-ubyte",
    "Datasets/MNIST/t10k-labels-idx1-ubyte"
]

trainExamplesToUse = 5000
testExamplesToUse = 10000

train_x = MNIST_readImages(paths[0], trainExamplesToUse)
train_y = oneHotEncode(MNIST_readLabels(paths[1], trainExamplesToUse), 10)
test_x = MNIST_readImages(paths[2], testExamplesToUse)
test_y = oneHotEncode(MNIST_readLabels(paths[3], testExamplesToUse), 10)

net = (
    Network(input_shape = 784, learning_rate=0.8)
        .dense(16)
        .sigmoid()
        .dense(32)
        .sigmoid()
        .dense(10)
        .softmax()
)
net.print_summary()
train_x = train_x / 255
test_x = test_x / 255

trainer = Trainer(net)
trainer.train(epochs = 100, 
              batch_size = 32, 
              train_x = train_x, 
              train_y = train_y, 
              cost = BinaryCrossEntropy(), 
              calcAccuracy = calcAccuracy)
print("The accuracy is", calcAccuracy(net.predict(test_x), test_y))
preds = np.argmax(net.predict(test_x), 1)
net.save("Models/MNIST(ANN)_1.npz")
for i in range(trainExamplesToUse):
    plt.imshow(np.reshape(test_x[i, :], (28, 28)))
    print(preds[i])
    plt.show()

