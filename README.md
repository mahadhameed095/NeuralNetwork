# NeuralNetwork
A library to construct ANNs and CNNs written using python.

## MNIST problem:
```
net = (
    Network(input_shape = 784, learning_rate=0.5)
        .dense(16)
        .sigmoid()
        .dense(16)
        .sigmoid()
        .dense(10)
        .softmax()
)

train_x = train_x / 255
test_x = test_x / 255

trainer = Trainer(net)
trainer.train(epochs, batch_size, train_x, train_y, BinaryCrossEntropy(), calcAccuracy)
print("The accuracy is", calcAccuracy(net.predict(test_x), test_y))
preds = np.argmax(net.predict(test_x), 1)
# net.save("MNIST")
for i in range(trainExamplesToUse):
    plt.imshow(np.reshape(test_x[i, :], (28, 28)))
    print(preds[i])
    plt.show()
```

## Convolutional MNIST:
```
net = (
    Network(input_shape = train_x.shape, learning_rate=0.03)
        .conv2d(num_kernels = 2, kernel_size = 6)
        .relu()
        .conv2d(num_kernels = 1, kernel_size = 6)
        .relu()
        .flatten()
        .dense(num_neurons = 10)
        .softmax()
)
net.print_summary()

# Training
trainer = Trainer(net)
trainer.train(epochs = 100, 
              batch_size = 64, 
              train_x = train_x, 
              train_y = train_y,
              cost = BinaryCrossEntropy(), 
              calcAccuracy= calcAccuracy)
              
net.save("Models/MNIST(CNN)_2.npz")
print("The accuracy on the test data =>", calcAccuracy(net.predict(test_x), test_y))
```
