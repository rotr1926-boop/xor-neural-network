# XOR Neural Network

Feedforward neural network that learns the XOR logic gate, implemented from scratch in C++.

## Architecture

- 2 inputs, 1 hidden layer (2 nodes), 1 output
- Sigmoid activation function
- Backpropagation with gradient descent

## Build & Run

```bash
g++ NeuralNetwork.c -o xor
./xor
```

Or to predict using trained weights:

```bash
g++ XorPredictor.c -o xor_predict
./xor_predict
```

## Training

The network is trained on the XOR truth table:

| Input | Output |
|-------|--------|
| 0 0   | 0      |
| 0 1   | 1      |
| 1 0   | 1      |
| 1 1   | 0      |

Weights are saved to `weights.txt` after training.
