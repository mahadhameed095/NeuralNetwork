import numpy as np
# from Network import Network
# from Trainer import Trainer
from Conv2d import Conv2d
import time
from Utils import getWindows

np.random.seed(1)
x = np.random.randint(1, 5, (128, 3, 28, 28)).astype(np.float32)
conv = Conv2d(x.shape, 5, 10, 0.1)
forward = conv.forward(x)
print(forward.shape)
dk, db, di = conv.backward(forward)

print("dk=",dk.shape)
print("db=",db.shape)
print("di=",di.shape)



# k = np.random.randint(0, 2, (2, 3, 3))
# print(x)
# print()
# print(k)
# print()
# print(correlate(x, k, 'valid').reshape(3, 3))

# from MeanSquaredError import MeanSquaredError

# x = np.array([[0, 0, 0, 0], 
#              [0, 1, 0, 1]], dtype = "float32")
# y = np.array([[0, 1, 1, 0]], dtype = "float32")

# net = Network()
# net.dense(2, 3, 0.2).tanh().dense(3, 1, 0.2).tanh()
# trainer = Trainer(net)
# trainer.train(100, x, y, MeanSquaredError())

# print(net.predict(x))
