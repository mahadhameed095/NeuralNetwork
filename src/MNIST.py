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
        return data.reshape(numToRead, 784).T  # resultant shape = (784, numToRed)


def MNIST_readLabels(path: str, numToRead: int) -> np.ndarray:
    with open(path, "rb") as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        assert magic_number == 2049, "Incorrect format file."
        num_labels = int.from_bytes(f.read(4), 'big')
        assert numToRead <= num_labels, "Maximum number of labels exceeded"
        return np.frombuffer(f.read(numToRead), dtype=np.uint8).reshape(1, numToRead)


# figure, ax = plt.subplots(figsize=(4,5))
# x = []
# y = []
# plt.ion()
# plot1, = ax.plot(x, y)
# plt.xlabel("Iterations",fontsize=18)
# plt.ylabel("Cost",fontsize=18)
# plt.show()
# def plotCost(iter, Error):
#     x.append(iter)
#     y.append(iter)
#     plot1.set_xdata(x)
#     plot1.set_ydata(y)
#     figure.canvas.draw()
#     figure.canvas.flush_events()
#     time.sleep(0.1)

trainExamplesToUse = 10000
testExamplesToUse = 10000
learning_rate = 0.5
epochs = 150
batch_size = 128
net = (
    Network(784)
        .dense(16, learning_rate)
        .sigmoid()
        .dense(16, learning_rate)
        .sigmoid()
        .dense(10, learning_rate)
        .softmax()
)

train_x = MNIST_readImages("../Datasets/MNIST/train-images-idx3-ubyte", trainExamplesToUse)
train_y = oneHotEncode(MNIST_readLabels("../Datasets/MNIST/train-labels-idx1-ubyte", trainExamplesToUse), 10)
test_x = MNIST_readImages("../Datasets/MNIST/t10k-images-idx3-ubyte", testExamplesToUse)
test_y = oneHotEncode(MNIST_readLabels("../Datasets/MNIST/t10k-labels-idx1-ubyte", testExamplesToUse), 10)
train_x = train_x / 255
test_x = test_x / 255


def calcAccuracy(forwardMat: np.ndarray, train_Y: np.ndarray):
    preds = np.argmax(forwardMat, 0)
    num_correct_pred = preds == np.argmax(train_Y, 0)
    num_correct_pred = np.count_nonzero(num_correct_pred)
    return (num_correct_pred) * 100 / train_Y.shape[1]


trainer = Trainer(net)
# trainer.train(epochs, batch_size, train_x, train_y, BinaryCrossEntropy(), calcAccuracy)
net.load("../Models/MNIST")
print("The accuracy is", calcAccuracy(net.predict(test_x), test_y))
preds = np.argmax(net.predict(test_x), 0)
# net.save("MNIST")
for i in range(trainExamplesToUse):
    plt.imshow(np.reshape(test_x[ :, i], (28, 28)))
    print(preds[i])
    plt.show()

