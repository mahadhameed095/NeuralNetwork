from pickletools import uint8
import numpy as np
from Network import Network
from Trainer import Trainer
from BinaryCrossEntropy import BinaryCrossEntropy
import matplotlib.pyplot as plt
import time
from Utils import oneHotEncode
def MNIST_readImages(path, numToRead) -> list:
    with open(path, "rb") as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        assert magic_number == 2051, "Incorrect format file." 
        num_images = int.from_bytes(f.read(4), 'big')
        assert numToRead <= num_images, "Maximum number of images exceeded"
        f.read(8) # skipping 2 integers
        data = np.frombuffer(f.read(784 * numToRead), dtype=np.uint8).astype(np.float32)
        return data.reshape(numToRead, 784).T # resultant shape = (784, numToRed)

def MNIST_readLabels(path, numToRead) -> list:
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

trainExamplesToUse = 2000

testExamplesToUse = 10000
learning_rate = 1
epochs = 100
batch_size = 2000
train_x = MNIST_readImages("datasets/MNIST/train-images-idx3-ubyte", trainExamplesToUse)
train_y = oneHotEncode(MNIST_readLabels("datasets/MNIST/train-labels-idx1-ubyte", trainExamplesToUse), 10)
train_x = train_x / 255
# test_x = np.asarray(MNIST_readImages("datasets/MNIST/t10k-images-idx3-ubyte", testExamplesToUse), dtype="float32")
# test_y = np.asarray(MNIST_readLabels("datasets/MNIST/t10k-labels-idx1-ubyte", testExamplesToUse), dtype = "float32")

net =   (
            Network()
               .dense(784, 16, learning_rate)
               .sigmoid()
               .dense(16, 16, learning_rate)
               .sigmoid()
               .dense(16, 10, learning_rate)
               .sigmoid()
        )

trainer = Trainer(net)
trainer.train(epochs, batch_size, train_x, train_y, BinaryCrossEntropy())    
preds = np.argmax(net.predict(train_x), 0,)
num_correct_pred = preds == np.argmax(train_y, 0)
num_correct_pred = np.count_nonzero(num_correct_pred)
print("Accuracy is", (num_correct_pred / trainExamplesToUse) * 100)

for i in range(trainExamplesToUse):
    plt.imshow(np.reshape(train_x[ :, i], (28, 28)))
    print(preds[i])
    print(train_y[:, i])
    plt.show()
    