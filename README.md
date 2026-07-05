# TinyNN

TinyNN is a lightweight, single-header neural network library written in C. It is built from scratch without external machine learning libraries and is intended for learning the fundamentals of neural networks and numerical optimization.

The library implements a fully connected feedforward neural network using matrix operations, sigmoid activation, numerical gradient estimation (finite differences), and gradient descent.

> **Note:** TinyNN is an educational project and currently uses finite differences instead of backpropagation for gradient computation.

## Features

* Single-header library (`tinynn.h`)
* Dynamic feedforward neural networks
* Configurable network architecture
* Matrix abstraction with basic linear algebra operations
* Sigmoid activation function
* Mean Squared Error (MSE) loss
* Gradient estimation using finite differences
* Gradient descent training
* No external dependencies

## Building

Compile using GCC or Clang.

```bash
gcc main.c -o tinynn -lm
```

Run:

```bash
./tinynn
```

## Creating a Neural Network

Define the network architecture as an array where each element represents the number of neurons in a layer.

```c
size_t arch[] = {2, 2, 1};

NN nn = nn_alloc(arch, ARRAY_LEN(arch));
NN grad = nn_alloc(arch, ARRAY_LEN(arch));
```

The above creates:

* Input layer: 2 neurons
* Hidden layer: 2 neurons
* Output layer: 1 neuron

## Initializing Parameters

Randomly initialize weights and biases.

```c
nn_rand(nn, 0.0f, 1.0f);
```

## Dataset Format

Training data is stored as a matrix.

Example XOR dataset:

```c
float td[] = {
    0,0,0,
    0,1,1,
    1,0,1,
    1,1,0,
};
```

Input matrix:

```text
0 0
0 1
1 0
1 1
```

Output matrix:

```text
0
1
1
0
```

## Training

```c
float eps  = 1e-1f;
float rate = 1e-1f;

for (size_t i = 0; i < 20000; i++) {
    nn_finite_diff(nn, grad, eps, ti, to);
    nn_learn(nn, grad, rate);
}
```

## Forward Inference

```c
MAT_AT(NN_INPUT(nn), 0, 0) = x1;
MAT_AT(NN_INPUT(nn), 0, 1) = x2;

nn_forward(nn);

float prediction = MAT_AT(NN_OUTPUT(nn), 0, 0);
```

## API Overview

### Matrix

| Function      | Description                    |
| ------------- | ------------------------------ |
| `mat_alloc()` | Allocate a matrix              |
| `mat_fill()`  | Fill matrix with a constant    |
| `mat_rand()`  | Fill matrix with random values |
| `mat_copy()`  | Copy one matrix into another   |
| `mat_dot()`   | Matrix multiplication          |
| `mat_sum()`   | Element-wise addition          |
| `mat_sig()`   | Apply sigmoid activation       |
| `mat_row()`   | Create a view of a matrix row  |
| `mat_print()` | Print a matrix                 |

### Neural Network

| Function           | Description                                 |
| ------------------ | ------------------------------------------- |
| `nn_alloc()`       | Create a neural network                     |
| `nn_rand()`        | Initialize weights and biases               |
| `nn_forward()`     | Perform forward propagation                 |
| `nn_cost()`        | Compute MSE loss                            |
| `nn_finite_diff()` | Estimate gradients using finite differences |
| `nn_learn()`       | Update parameters using gradient descent    |
| `nn_print()`       | Print network parameters                    |

