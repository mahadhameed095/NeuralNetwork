import numpy as np
from Network import Network
from Trainer import Trainer
from BinaryCrossEntropy import BinaryCrossEntropy
import matplotlib.pyplot as plt
from Utils import oneHotEncode

def MNIST_readImages(path: str, numToRead: int) -> np.ndarray:
    with open(path, "rb") as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        assert magic_number == 2051, "Incorrect format file."
        num_images = int.from_bytes(f.read(4), 'big')
        assert numToRead <= num_images, "Maximum number of images exceeded"
        f.read(8)  # skipping 2 integers
        data = np.frombuffer(f.read(784 * numToRead), dtype=np.uint8).astype(np.float32)
        return data.reshape(numToRead, 28, 28)


def MNIST_readLabels(path: str, numToRead: int) -> np.ndarray:
    with open(path, "rb") as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        assert magic_number == 2049, "Incorrect format file."
        num_labels = int.from_bytes(f.read(4), 'big')
        assert numToRead <= num_labels, "Maximum number of labels exceeded"
        return np.frombuffer(f.read(numToRead), dtype=np.uint8).reshape(1, numToRead)


trainExamplesToUse = 1000
testExamplesToUse = 10000
paths = [
    "Datasets/MNIST/train-images-idx3-ubyte",
    "Datasets/MNIST/train-labels-idx1-ubyte",
    "Datasets/MNIST/t10k-images-idx3-ubyte",
    "Datasets/MNIST/t10k-labels-idx1-ubyte"
]

#Reading the data
train_x = MNIST_readImages(paths[0], trainExamplesToUse)
train_y = oneHotEncode(MNIST_readLabels(paths[1], trainExamplesToUse), 10)
test_x = MNIST_readImages(paths[2], testExamplesToUse)
test_y = oneHotEncode(MNIST_readLabels(paths[3], testExamplesToUse), 10)

#Feature scaling
train_x = train_x / 255
test_x = test_x / 255


train_x = train_x[:, np.newaxis, ...]
test_x = test_x[:, np.newaxis, ...]

#Network configuration
net = (
    # 441
    Network()
        .conv2d(num_kernels = 3, kernel_size = 5)
        .relu()
        .conv2d(num_kernels = 1, kernel_size = 5)
        .relu()
        .flatten()
        .dense(num_neurons = 10)
        .softmax()
)


def calcAccuracy(forwardMat: np.ndarray, train_Y: np.ndarray):
    preds = np.argmax(forwardMat, 0)
    num_correct_pred = preds == np.argmax(train_Y, 0)
    num_correct_pred = np.count_nonzero(num_correct_pred)
    return num_correct_pred * 100 / train_Y.shape[1]

#Training

hyper_parameters = {
    "input_shape" : train_x.shape,
    "learning_rate" : 0.01,
    "epochs" : 50,
    "batch_size" : 32
}

trainer = Trainer(net)
trainer.train(hyper_parameters, train_x, train_y, BinaryCrossEntropy(), calcAccuracy)

# net.load("../Models/ConvMNIST")
# preds = np.argmax(net.predict(test_x), 0)

print(calcAccuracy(net.predict(test_x), test_y))
net.save("../Models/ConvMNIST2")
# for i in range(int(trainExamplesToUse)):
#     plt.imshow(test_x[i].reshape(28, 28))
#     # print(preds[i])
#     plt.show()
