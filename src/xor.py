import numpy as np
from Network import Network
from Trainer import Trainer
from MeanSquaredError import MeanSquaredError

x = np.array([[0, 0, 0, 0], 
             [0, 1, 0, 1]], dtype = "float32")
y = np.array([[0, 1, 1, 0]], dtype = "float32")

net = Network()
net.dense(2, 3, 0.2).tanh().dense(3, 1, 0.2).tanh()
trainer = Trainer(net)
trainer.train(100, x, y, MeanSquaredError())

print(net.predict(x))
