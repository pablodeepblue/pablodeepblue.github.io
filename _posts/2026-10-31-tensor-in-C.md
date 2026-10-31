---
layout: post
title: "Tensor In C"
date: 2026-10-31
categories: tutorials
---

Do you actually know how Neural Nets and Transformers are implemented? Like actually? It wouldn't surprise me if you didn't. Its a hard system and its often hard to get your head around all the complex parts. Don't worry though. In this "tutorial" I shall teach you how to implement everything from scratch using python. However I am going to do this a bit different than everyone else. A common complaint I hear when people use python to implement Neural Neural nets or the such is "from outside import *". Basically that using libs is not "starting from scratch". While I find it a bit pedantic I decided to say, "Okay then".

This will be a EXTREMELY simplified overview of Transformers and Tensors. Like insanely simplified using only Python and, later on, C. Production integration of C and production standards for python wild differently, so don't take this as a actual tutorial to make a lib.

If you want to see a actual implementation that uses CUDA and the such look at the repo for [pytorch](https://github.com/pytorch/pytorch).

This project is a work in progress. By the time I post this blog I will have finished until attention heads, though it may take a bit for this blog to catch up. See my work here [Tensor in C](https://github.com/pablodeepblue/tensor_in_c).

# Part 1: Neural Net from Scratch

Lets start with the most basic unit of any of these. A nueron.

## Neuron

The formula for a neuron is the following.

$$
N(x) = \sigma(w \cdot x + b)
$$

Now, what do these things mean? Lets start with $x$. It is a input. Typically a vector, though it could in practice be a number as well.
For our purposes we will assume the $x$ is always a vector.

$w$ is the "weight" of the neuron, a vector of the same size as $x$. In essence how much it modifies the original $x$ via dot product.

$$
\text{dot product: } a \cdot b = \sum_{\text{for } i \in range(|a|)} a_i b_i
$$

The $b$ is the bias. Since the output of the dotproduct is number, b is a number as well.

Finally, $\sigma$ is a activator function. Why do we need it? Many sources could explain in more detail, but the long and short of it is that without it all neurons would be linear. Why is this a problem? Most complex problems are not linear. Being non-linear allow the the neural net to actually model these problems. It will get more obvious down the road.

$w$ and $b$ are like dials, where via a series of training steps we make the neuron do what we want it to do.

Here is a simple implementation

```py
class Neuron:
    def __init__(self, input_size):
        self.weights = [random.gauss(0, 1)*0.1 for _ in range(input_size)]
        self.input_size = input_size
        self.bias = 0
        self.last_input = None

    @staticmethod
    def dot_product(vector1, vector2) -> float:
        assert len(vector1) == len(vector2)
        return float(sum(x*y for x,y in zip(vector1, vector2)))

    def forward(self, X):
        self.last_input = list(X)
        return self.dot_product(self.weights, self.last_input) + self.bias

    def backward(self, upstream_gradient, learning_rate):
        assert self.last_input is not None, "forward must be called before backward"
        input_gradient = [upstream_gradient * weight for weight in self.weights]
        for i, value in enumerate(self.last_input):
            self.weights[i] -= learning_rate * upstream_gradient * value
        self.bias -= learning_rate * upstream_gradient
        return input_gradient

    #call just lets us call the forward when using the instance like a class 
    #(pytorch does something similar)
    def __call__(self, X):
        return self.forward(X)
```

We will add the activator function later.

"Hey" you may say "whats the backward function?".

I'm glad you asked.

Backward is what will we use for backpropagation. What is backpropagation? As IBM put it.

> Backpropagation is a machine learning technique essential to the optimization of artificial neural networks. It facilitates the use of gradient descent algorithms to update network weights, which is how the deep learning models driving modern artificial intelligence (AI) “learn.”

This is what we use to make AI learn. Gradients (from calc). We pass data, we see how much we are wrong by (error) and then use the error to adjust the weights and biases of neuron (the dials).

pytorch has more sophisticated ways of modeling this via complicated graphs that allow for automatically calculating of the gradients and updating them upon calculating the loss. We will be doing manually to show its essence.

You can look it over yourself in the code but if you look into it is simply

1. Find the gradient you are backproping (upstream_gradient * weights)
2. update your weights and your bias using the upstream_gradient (weights require the last input for this) (learning rate modifies how much of this gradient the network should learn by)
3. return the backprop to continue backproping.

For now most of the gradient stuff is easy to calculate (since its mostly just linear equations) but later down the line it gets a bit harder.

## Layer

A layer is a stack neurons. Its as simple as that. By stacking the neurons we can turn the vector to number function to vector to vector. This behavior allows us to stack multiple layers so we can have multiple different neurons learning different things. Output size is equal to the number of neurons stacked.

```Python
class Layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.neurons = [
            Neuron(input_size)
            for _ in range(output_size)
        ]

    def append(self, neuron:Neuron) -> None:
        assert isinstance(neuron, Neuron), "neuron is not a Neuron"
        assert self.input_size is None or self.input_size == neuron.input_size
        self.neurons.append(neuron)
        self.output_size += 1

    def forward(self, X):
        X = list(X)
        out = [
            neuron(X)
            for neuron in self.neurons
        ]
        return out

    def backward(self, upstream_gradient, learning_rate):
        assert len(upstream_gradient) == self.output_size, "mismatch gradient and layer size"
        input_gradient = [0.0] * self.input_size
        for neuron, gradient in zip(self.neurons, upstream_gradient):
            neuron_input_gradient = neuron.backward(gradient, learning_rate)
            for i, value in enumerate(neuron_input_gradient):
                input_gradient[i] += value
        return input_gradient
    
    def __call__(self, X):
        return self.forward(X)
```

The backprop is pretty simple here as well, just sum all the gradients of the neurons and send it up.

## Activator

Now we implement the activator. Why here? Well it may be a consequence of using pytorch too much but I prefer to have the activator as its own separate "layer". It makes swapping it out easier and I feel its easier to understand.

```Python
def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0

class Activator:
    def __init__(self, func, derivative):
        self.func = func
        self.derivative = derivative

    def forward(self, X):
        self.last_input = list(X)
        return list(map(self.func, self.last_input))
    
    def backward(self, upstream_gradient, learning_rate):
            out = [
                grad * self.derivative(z)
                for grad, z in zip(upstream_gradient, self.last_input)
            ]
            return out
    
    def copy(self):
        return Activator(self.func, self.derivative)

    def __call__(self, X):
        return self.forward(X)

relu_activator = Activator(relu, relu_derivative)
```

In the actual repo I implemented a bunch but here I will only be using ReLU, one of the most common activator function.

## Softmax

Softmax goes at the end of a neural net to turn all the values into a probability.

```Python
class Softmax:
    def forward(self, X):
        Z = list(map(exp, X))
        s = sum(Z)
        return [z/s for z in Z]
    
    def __call__(self, X):
        return self.forward(X)
```

Math Equation

$$
\operatorname{softmax}(z_i) =

\frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
$$

When training a classifier, this is often used to figure out what the actual prob for each classified value is.

## Neural Net

Finally we get to the neural net. Not much to say here in terms of structure. What a layer is to neurons, a Neural Net is to layers.

```py
class Neural_Net:
    def __init__(self, input_size, *hidden_layers, output_size=None, activation_func=None):
        self.last_input = None
        self.last_output = None
        self.net = []
        self.output_func = Softmax()
        if activation_func is None:
            activation_func = relu_activator
        sizes = [input_size] + list(hidden_layers) + ([output_size] if output_size else [])
        for i, (inp, oup) in enumerate(zip(sizes, sizes[1:])):
            self.net.append(Layer(inp, oup))
            if i == len(sizes) - 2:
                pass
            else:
                self.net.append(activation_func.copy())
    
    def forward(self, X):
        self.last_input = list(X)
        y = X
        for layer in self.net:
            y = layer(y)
        self.last_output = self.output_func(y)
        return self.last_output
    
    def backward(self, true, learning_rate):
        grad = [
            (out - true_per)
            for out, true_per in zip(self.last_output, true)
        ] 
        for layer in self.net[::-1]:
            grad = layer.backward(grad, learning_rate)

    def __call__(self, X):
        return self.forward(X)
```

At this point though, the neural net is the initiaton of the backprop chain. Here we use the true value (often a identity vector) to calculate how wrong we are. Using this info we can adjust all the neurons to match.

## Training

Okay I fibbed a little. We will be using SOME external libs. Loading data from datasets is more work that its worth so we will be importing out dataset to train out data.

For context, the dataset is handwritten numbers and we will be using this to classify them 0 to 9.

The eye func turns a index into a 1 hot vector (a binary array (or vector) of N elements (where N is the total number of categories, classes, or vocabulary words) used to represent categorical data) to show that the prob that it is 7 for example, is exactly one and we want the NN to target this correct value.

```Python
def eye(index, shape):
    return [1 if i == index else 0 for i in range(shape)]

def argmax(l1):
    m = l1[0]
    i = 0
    for j, x in enumerate(l1):
        if x > m:
            i = j
            m = x
    return i
    
if __name__ == "__main__":
    random.seed(42)
    from sklearn.datasets import load_digits
    from time import perf_counter
    X,_y = load_digits(return_X_y=True); X = X / 16
    Y = [eye(obj, 10) for obj in _y]
    network = Neural_Net(64, 32, 10)
    start = perf_counter()
    results = [argmax(network(x)) == y for x,y in zip(X,_y)]
    print(f"prelearning {sum(results)/len(results)}")
    for epoch in range(300):
        #print(f"epoch {epoch}")
        for x,y in zip(X,Y):
            network(x)
            network.backward(y, 0.5)
    end = perf_counter()
    results = [argmax(network(x)) == y for x,y in zip(X,_y)]
    print(f"postlearning {sum(results)/len(results)}; time {end - start}")
```

And thats it. Here is a neural net. Super simple. Most of the hard part of neural nets is just the backprop implementation to learn, but besides that it is just abstracting. Libraries like pytorch make this super easy by allowing you not to worry about back prop or the nitty gritty.

## 2: Improving the layer

If you ever done a ML class, you probably seen this equation:

$$
N(x) = \sigma(W \cdot x + b)
$$

What is this? Well, when computing a layer it's kind of wasteful to go neuron to neuron so instead we use matrix multiplication.

$$
\mathbf{y} = \mathbf{A}\mathbf{x}
$$

$$
y_i = A_{i} \cdot x
$$

Where $A_i$ is a row in a matrix.

Basically the same thing as before but the benefit of doing this way is that we can compute everything in one go rather than going neuron to neuron. Matrix Mults are also incredibly easy to paralize using GPUs so it is the implementain used for all AI.

To this end we will modify the original to use this instead.

```Python
class Layer:
    def __init__(self, input_size, output_size):
        self.weight = [[random.gauss(0, 1)*0.1 for _ in range(input_size)] for _ in range(output_size)]
        self.bias = [0 for _ in range(output_size)]
        self.input_size = input_size
        self.output_size = output_size
        self.last_input = None

    @staticmethod
    def dot_product(vector1, vector2) -> float:
        assert len(vector1) == len(vector2)
        return float(sum(x*y for x,y in zip(vector1, vector2)))
    
    def forward(self, X):
        # reimplemented as self contained for loop. Cool, huh?
        self.last_input = X
        output = [
            self.dot_product(neuron, X) + self.bias[i]
            for i, neuron in enumerate(self.weight)
        ]
        return output
    
    # we compressed the backprop so now its just a simple as 1 and 2
    def backward(self, upstream_gradient, learning_rate):
        assert len(upstream_gradient) == self.output_size, "mismatch gradient and layer size"
        input_gradient = [0.0] * self.input_size
        for i, (neuron, gradient) in enumerate(zip(self.weight, upstream_gradient)):
            # calc input grad
            neuron_input_gradient = [ gradient * value for value in neuron ]
            for j, value in enumerate(neuron_input_gradient):
                input_gradient[j] += value

            #propagate
            step = learning_rate * gradient
            for j, value in enumerate(self.last_input):
                neuron[j] -= step * value
            self.bias[i] -= step
            
        return input_gradient

    def __call__(self, X):
        return self.forward(X)
```

The backprop is just a refactored form of the original backprop.

If you use the previous test you get much faster due to less dependancies.

## 3 Adding batches/sequnces.



For latter parts, it will not be enough to only take one input at a time. We will instead be using batches, a list of input vectors. When we implement this every input is independant. However, latter down the line we will be using this to have sequential data, data where order matters. More on that later.

### Activator

```Python
def copy_md_list(X):
    return [copy_md_list(x) if isinstance(x, list) else x for x in X]

class Activator:
    def __init__(self, func, derivative):
        self.func = func
        self.derivative = derivative
        self.last_input_batch = None

    # now takes multidim

    def forward(self, X):
        self.last_input_batch = copy_md_list(X)
        return [list(map(self.func, last_input)) for last_input in self.last_input_batch]
    
    def backward(self, upstream_gradient_batch, learning_rate):
            out = [
                [
                    grad * self.derivative(z)
                    for grad, z in zip(upstream_gradient, last_input)
                ]
                for last_input, upstream_gradient in zip(self.last_input_batch, upstream_gradient_batch)
            ]
            return out
    
    def copy(self):
        return Activator(self.func, self.derivative)

    def __call__(self, X):
        return self.forward(X)
```

### Layer
```Python
class Layer:
    def __init__(self, input_size, output_size):
        self.weight = [[random.gauss(0, 1)*0.1 for _ in range(input_size)] for _ in range(output_size)]
        self.bias = [0 for _ in range(output_size)]
        self.input_size = input_size
        self.output_size = output_size
        self.last_input_batch = None

    @staticmethod
    def dot_product(vector1, vector2) -> float:
        assert len(vector1) == len(vector2)
        return float(sum(x*y for x,y in zip(vector1, vector2)))
    
    @staticmethod
    def linear(input_, weight, bias):
        return [
            [
                Layer.dot_product(row, vector) + bias[i]
                for i, row in enumerate(weight)
            ] for vector in input_
        ]
    
    def forward(self, X):
        # reimplemented as self contained for loop. Cool, huh?
        self.last_input_batch = copy_md_list(X)
        return self.linear(X, self.weight, self.bias)
    
    # we compressed the backprop so now its just a simple as 1 and 2
    def backward(self, upstream_gradient_batch, learning_rate): 
        assert len(upstream_gradient_batch[0]) == self.output_size, "mismatch gradient and layer size"
        w_grad = [[0.0]*self.input_size for _ in range(self.output_size)]
        b_grad = [0.0]*self.output_size
        input_gradient_batch = []
        for last_input, upstream in zip(self.last_input_batch, upstream_gradient_batch):
            input_gradient = [0.0]*self.input_size
            for i, (neuron, g) in enumerate(zip(self.weight, upstream)):
                for j, w in enumerate(neuron):
                    input_gradient[j] += g * w          # uses the OLD weights
                    w_grad[i][j] += g * last_input[j]
                b_grad[i] += g
            input_gradient_batch.append(input_gradient)

        
        n = len(upstream_gradient_batch)
        for i in range(self.output_size):
            for j in range(self.input_size):
                self.weight[i][j] -= learning_rate * w_grad[i][j] / n
            self.bias[i] -= learning_rate * b_grad[i] / n
        return input_gradient_batch

    def __call__(self, X):
        return self.forward(X)
```

### Softmax

```Python
class Softmax:
    def forward(self, X):
        out_batch = []
        for x in X:
            Z = list(map(exp, x))
            s = sum(Z)
            out_batch.append([z/s for z in Z])
        return out_batch
    
    def __call__(self, X):
        return self.forward(X)
```

### Neural Net

```Python
class Neural_Net:
    def __init__(self, input_size, *hidden_layers, output_size=None, activation_func=None):
        self.last_input_batch = None
        self.last_output_batch = None
        self.net = []
        self.output_func = Softmax()
        if activation_func is None:
            activation_func = relu_activator
        sizes = [input_size] + list(hidden_layers) + ([output_size] if output_size else [])
        for i, (inp, oup) in enumerate(zip(sizes, sizes[1:])):
            self.net.append(Layer(inp, oup))
            if i == len(sizes) - 2:
                pass
            else:
                self.net.append(activation_func.copy())
    
    def forward(self, X):
        self.last_input_batch = copy_md_list(X)
        y = X
        for layer in self.net:
            y = layer(y)
        self.last_output_batch = self.output_func(y)
        return self.last_output_batch
    
    def backward(self, true_batch, learning_rate):
        grad = [
            [
                (out - true_per)
                for out, true_per in zip(last_output, true)
            ] for true, last_output, last_input in zip(true_batch, self.last_output_batch, self.last_input_batch)
        ] 
        for layer in self.net[::-1]:
            grad = layer.backward(grad, learning_rate)

    def __call__(self, X):
        return self.forward(X)
```

We can now also train in batched mode

```Python
def get_acc(network, X, Y):
    return [argmax(y_) == y for y_, y in zip(network(X), Y)]

if __name__ == "__main__":
    random.seed(42)
    from sklearn.datasets import load_digits
    from time import perf_counter
    X,_y = load_digits(return_X_y=True); X = (X / 16).tolist()
    Y = [eye(obj, 10) for obj in _y]
    network = Neural_Net(64, 32, 10)
    start = perf_counter()
    results = get_acc(network, X, _y)
    print(f"prelearning {sum(results)/len(results)}")
    for epoch in range(300):
        #print(f"epoch: {epoch}")
        network(X)
        network.backward(Y, 0.5)
    end = perf_counter()
    results = get_acc(network, X, _y)
    print(f"postlearning {sum(results)/len(results)}; time {end - start}")
```

in the end we sacrifice some model performance but we do get benefits in actual perf

(Finished to here by 10-31-2026)

# 4: Attention (to be done)
