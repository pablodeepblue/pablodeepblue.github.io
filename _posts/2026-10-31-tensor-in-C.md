---
layout: post
title: "Tensor In C"
date: 2026-10-31
categories: tutorials
---

Do you actually know how Neural Nets and Transformers are implemented? Like actually? It wouldn't surprise me if you didn't. It's a hard system, and it's often hard to get your head around all the complex parts. Don't worry though. In this "tutorial" I shall teach you how to implement everything from scratch using Python. However, I am going to do this a bit differently than everyone else. A common complaint I hear when people use Python to implement neural nets and the like is "from outside import *". Basically, that using libs is not "starting from scratch". While I find it a bit pedantic, I decided to say, "Okay then".

This will be an EXTREMELY simplified overview of Transformers and Tensors. Like insanely simplified, using only Python and, later on, C. Production integration of C and production standards for Python differ wildly, so don't take this as an actual tutorial for making a lib. (A tensor, by the way, is just a general name for a grid of numbers: a single number, a vector, a matrix, or a stack of matrices. Everything in this tutorial boils down to operating on these.)

If you want to see an actual implementation that uses CUDA and such, look at the repo for [pytorch](https://github.com/pytorch/pytorch).

This project is a work in progress. By the time I post this blog I will have finished up to attention heads, though it may take a bit for this blog to catch up. See my work here: [Tensor in C](https://github.com/pablodeepblue/tensor_in_c).

# Part 1: Neural Net from Scratch

Let's start with the most basic unit of any of these: a neuron.

## Neuron

The formula for a neuron is the following.

$$
N(x) = \sigma(w \cdot x + b)
$$

Now, what do these things mean? Let's start with $x$. It is an input. Typically it is a vector, though it could in practice be a number as well.
For our purposes we will assume that $x$ is always a vector.

$w$ is the "weight" of the neuron, a vector of the same size as $x$. In essence, it is how much each part of the original $x$ is scaled before everything is added up, which is exactly what the dot product does.

$$
\text{dot product: } a \cdot b = \sum_{\text{for } i \in range(|a|)} a_i b_i
$$

The $b$ is the bias. Since the output of the dot product is a number, $b$ is a number as well. It lets the neuron shift its output up or down regardless of the input.

Finally, $\sigma$ is an activator function. Why do we need it? Many sources could explain in more detail, but the long and short of it is that without it all neurons would be linear, and stacking linear functions on top of each other just gives you another linear function. Why is this a problem? Most complex problems are not linear. Being non-linear allows the neural net to actually model these problems. It will get more obvious down the road.

$w$ and $b$ are like dials, where via a series of training steps we make the neuron do what we want it to do.

Here is a simple implementation.

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

The weights start as small random numbers so that different neurons don't all learn the exact same thing, and the bias starts at zero.

We will add the activator function later.

"Hey," you may say, "what's the backward function?"

I'm glad you asked.

Backward is what we will use for backpropagation. What is backpropagation? As IBM put it:

> Backpropagation is a machine learning technique essential to the optimization of artificial neural networks. It facilitates the use of gradient descent algorithms to update network weights, which is how the deep learning models driving modern artificial intelligence (AI) “learn.”

This is what we use to make AI learn. Gradients (from calculus) tell us which direction, and by how much, each dial should be turned to make the error smaller. We pass data through, we see how much we are wrong by (the error), and then we use the error to adjust the weights and biases of the neuron (the dials). Moving the dials a little bit in the direction that reduces the error is called gradient descent.

pytorch has more sophisticated ways of modeling this via complicated graphs that allow for automatically calculating the gradients and updating them upon calculating the loss. We will be doing it manually to show its essence.

You can look it over yourself in the code, but it is simply:

1. Find the gradient you are backpropping (upstream_gradient * weights). This is the chain rule from calculus: the gradient arriving from the next step, multiplied by how much this neuron's output depends on each input.
2. Update your weights and your bias using the upstream_gradient (weights require the last input for this, which is why forward saves it). The learning rate modifies how much of this gradient the network should learn by. Too big and training becomes unstable, too small and it learns very slowly.
3. Return the gradient so the backprop can continue to the previous step.

For now most of the gradient math is easy to calculate (since it's mostly just linear equations), but later down the line it gets a bit harder.

## Layer

A layer is a stack of neurons. It's as simple as that. By stacking the neurons we can turn the vector-to-number function into a vector-to-vector function. This behavior allows us to stack multiple layers so we can have multiple different neurons learning different things. The output size is equal to the number of neurons stacked.

```py
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

The backprop is pretty simple here as well: just sum all the gradients of the neurons and send it up. We sum because every input feeds into every neuron in the layer, so each input affects the error through all of them.

## Activator

Now we implement the activator. Why here? Well, it may be a consequence of using pytorch too much, but I prefer to have the activator as its own separate "layer". It makes swapping it out easier and I feel it's easier to understand.

```py
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

In the actual repo I implemented a bunch, but here I will only be using ReLU, one of the most common activator functions. ReLU simply passes positive numbers through unchanged and turns negative numbers into zero. Its derivative is therefore 1 for positive inputs and 0 otherwise. Since the activator has no weights of its own, its backward step only scales the gradient and has nothing to update (which is why it ignores the learning rate).

## Softmax

Softmax goes at the end of a neural net to turn all the values into probabilities.

```py
def exp(value):
    return 2.718281828459045 ** value

class Softmax:
    def forward(self, X):
        Z = list(map(exp, X))
        s = sum(Z)
        return [z/s for z in Z]
    
    def __call__(self, X):
        return self.forward(X)
```

Math equation:

$$
\operatorname{softmax}(z_i) =
\frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
$$

Every output is between 0 and 1, and all the outputs add up to 1, so they can be read as probabilities. When training a classifier, this is often used to figure out what the actual probability for each class is. Note that this simple version can overflow if the inputs get very large; production implementations subtract the largest value first to avoid this, but we will keep it simple.

## Neural Net

Finally we get to the neural net. There's not much to say here in terms of structure. What a layer is to neurons, a neural net is to layers.

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

The sizes work like this: `Neural_Net(64, 32, 10)` means 64 inputs, one hidden layer of 32 neurons, and 10 outputs. Every layer except the last is followed by an activator. The last layer is followed by the softmax instead, which is applied separately in forward.

At this point, though, the neural net is the initiation of the backprop chain. Here we use the true value (often a one-hot vector, see below) to calculate how wrong we are. Using this info we can adjust all the neurons to match. The starting gradient is simply the output probabilities minus the true values. This is not arbitrary: it is what you get when you work out the derivative of softmax combined with the cross-entropy loss, a common way to measure the error of a classifier, and it happens to simplify very nicely.

## Training

Okay, I fibbed a little. We will be using SOME external libs. Loading data from datasets is more work than it's worth, so we will be importing our dataset to train our network.

For context, the dataset is handwritten digits (scikit-learn's `load_digits`, which has 1,797 small 8x8 grayscale images, giving us 64 inputs each), and we will be using it to classify them 0 to 9. Each pixel value goes from 0 to 16, which is why we divide by 16 to get numbers between 0 and 1.

The eye func turns an index into a one-hot vector (a binary array (or vector) of N elements (where N is the total number of categories, classes, or vocabulary words) used to represent categorical data) to show that the probability that it is 7, for example, is exactly one, and we want the NN to target this correct value. The argmax func does the opposite: it finds the index of the largest value, which is the digit the network thinks is most likely.

```py
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

We check the accuracy before training (it should be around chance, about 10%) and after. An epoch is one full pass over the dataset, and here we update the weights after every single example. For simplicity we also measure accuracy on the same data we trained on; a real project would hold some data back for testing.

And that's it. Here is a neural net. Super simple. Most of the hard part of neural nets is just the backprop implementation to learn, but besides that it is just abstracting. Libraries like pytorch make this super easy by allowing you not to worry about backprop or the nitty-gritty.

## 2: Improving the layer

If you have ever taken an ML class, you have probably seen this equation:

$$
N(x) = \sigma(W \cdot x + b)
$$

What is this? Well, when computing a layer it's kind of wasteful to go neuron by neuron, so instead we use matrix multiplication.

$$
\mathbf{y} = \mathbf{A}\mathbf{x}
$$

$$
y_i = A_{i} \cdot x
$$

Where $A_i$ is a row in a matrix.

Basically it's the same thing as before, but the benefit of doing it this way is that we can compute everything in one go rather than going neuron by neuron. Matrix mults are also incredibly easy to parallelize using GPUs, so it is the implementation used for all AI. (Our Python version below still loops over the rows one at a time, so the real speedups will come later when we move to C.)

To this end we will modify the original to use this instead. Each row of the weight matrix is one neuron's weights, and the bias is now one list for the whole layer.

```py
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

If you use the previous test, you get much faster results due to fewer dependencies (there are no longer separate neuron objects and method calls for every single neuron).

## 3: Adding batches/sequences

For later parts, it will not be enough to only take one input at a time. We will instead be using batches, a list of input vectors. When we implement this, every input is independent. However, later down the line we will be using this to handle sequential data, which is data where order matters. More on that later.

### Activator

```py
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
```py
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

The big change in the layer is the backward pass. Instead of updating the weights after every example, we add up the gradients from the whole batch, average them (the division by `n`), and only then update the weights once. That is also why the input gradients are computed with the old weights: all the examples in the batch must be judged against the same weights.

### Softmax

```py
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

```py
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

We can now also train in batched mode.

```py
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

In the end we sacrifice some model performance, but we do get benefits in actual performance. The model learns less because it now makes only one weight update per epoch (here the whole dataset is a single batch) instead of one update per example, but each epoch runs much faster since the work is done in one pass.

(Finished to here by 10-31-2026)

# 4: Attention (to be done)