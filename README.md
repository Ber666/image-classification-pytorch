### CNN for Image Classification with PyTorch

##### Requirement

Python 3.7

PyTorch(Stable) 1.4

numpy 1.18.1

matplotlib 3.1.2

##### Usage

First, download and unzip MNIST dataset into the dictionary `dataset`.

Then run the following command line:

```bash
python main.py
```

This command will start to train the model on the train set using GPU(if available) or CPU, and save the model to `checkpoint.pth` each epoch.

```bash
python test.py
```

This command will test the trained model on the test set.

Also, you can run `parse_dataset.py` to play with the dataset, to show some pictures and analyze the problem your model made.

##### Net Architecture

Conv2d(4 output channels, kernelsize=5)
ReLU
MaxPool2d(kernelsize=2, stride=2)
Conv2d(4 output channels, kernelsize=5)
ReLU
MaxPool2d(kernelsize=2, stride=2)
FC layer(10 output unit)

Cross-entropy as loss function

##### Experiment

On MNIST dataset, after training for 100 epochs, we got the accuracy of 97.84%