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

# Part 2: Improving the layer

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

# Part 3: Adding batches/sequences

For later parts, it will not be enough to only take one input at a time. We will instead be using batches, a list of input vectors. When we implement this, every input is independent. However, later down the line we will be using this to handle sequential data, which is data where order matters. More on that later.

## Activator

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

## Layer
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

## Softmax

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

## Neural Net

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

# Part 4: Attention & Transformer Block

In the last part, every vector in a batch was independent. That is useful for images: one image does not need to know anything about the image next to it. Language is different. The meaning of a word often depends on the words around it. In the sentence "the cat sat because it was tired," the word "it" needs to look back at the earlier words to figure out what it means.

Attention is how a transformer lets a token look at the other tokens in its sequence. The data still looks like the batch from the last part: a list of vectors. We are changing what each row means. Instead of one row being one unrelated example, one row is now one token (a word, or part of a word) in the same piece of text.

We will still call the whole thing `X`, but its shape is now:

$$
X = [\text{sequence length}, \text{vector size}]
$$

For example, a sequence with 6 tokens where every token has 4 numbers has shape `[6, 4]`. To keep the implementation small, we handle one sequence at a time here. A production model adds an outer batch dimension so it can process many sequences at once. By the end of this part, every token will have a new vector that includes information from the earlier tokens it chose to pay attention to.

## Query, Key, and Value

We start by giving every token three different versions of itself:

$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$

`Q` stands for query, `K` for key, and `V` for value. You can think of the query as what a token is looking for, the key as what a token advertises about itself, and the value as the information it has to give away. `W_Q`, `W_K`, and `W_V` are three learned layers, like the ones we already made.

For every query, we take a dot product with every key. A larger dot product means the query and key are a better match. Then we divide by $\sqrt{d_k}$, where $d_k$ is the size of the key vector. This keeps the values from getting too large as the vector size grows.

$$
S = \frac{QK^T}{\sqrt{d_k}}
$$

The transpose on $K$ is there so every query can be compared with every key. `S` is a square matrix. Row $i$ says how much token $i$ should pay attention to every token in the sequence.

This is causal attention, which means a token may look at itself and earlier tokens, but never at tokens that come after it. If a language model could see future words while trying to predict the next word, it would be cheating. We enforce this by replacing the upper half of the score matrix with negative infinity. Softmax turns those values into zero.

Before writing that out, we pull a few small operations out of `Layer` so we can call them directly. `linear` is the matrix multiplication from the last part. Its bias is optional because the final weighted average in attention does not need one.

```py
def sqrt(value):
    return value ** 0.5

def dot_product(vector1, vector2):
    assert len(vector1) == len(vector2)
    return sum(x * y for x, y in zip(vector1, vector2))

def transpose(matrix):
    return [list(row) for row in zip(*matrix)]

def sum_2d(*matrices):
    return [
        [sum(values) for values in zip(*rows)]
        for rows in zip(*matrices)
    ]

def linear(input_, weight, bias=None):
    return [
        [
            dot_product(row, vector) + (0 if bias is None else bias[i])
            for i, row in enumerate(weight)
        ]
        for vector in input_
    ]

def softmax(X):
    output = []
    for row in X:
        values = [exp(value) for value in row]
        total = sum(values)
        output.append([value / total for value in values])
    return output
```

Now we can make the score matrix and apply the causal mask:

```py
def attention_linear(key, query):
    dk = sqrt(len(key[0]))
    return [
        [
            dot_product(q, k) / dk if j <= i else float('-inf')
            for j, k in enumerate(key)
        ]
        for i, q in enumerate(query)
    ]
```

For a sequence of four tokens, the mask looks like this. The `X`s are positions the token is not allowed to see.

$$
\begin{bmatrix}
\checkmark & X & X & X \\
\checkmark & \checkmark & X & X \\
\checkmark & \checkmark & \checkmark & X \\
\checkmark & \checkmark & \checkmark & \checkmark
\end{bmatrix}
$$

We apply softmax to each row of scores. That makes each row add up to one, so it becomes a list of how much attention each earlier token gets. Finally, we use those attention weights to take a weighted average of the values.

$$
A = \operatorname{softmax}(S)
$$

$$
\operatorname{Attention}(Q, K, V) = AV
$$

If a row of `A` puts 80% of its weight on one value and 20% on another, its output is mostly the first value with a little of the second mixed in. This is the entire idea of attention: build a new representation for each token from the information in the tokens it decided matters.

## One Head

Here is the forward pass for one attention head. The `Layer` class is the batched layer from the last part. Notice that we use the same `X` for queries, keys, and values. That is called self-attention. Attention can also use different inputs for them, but transformers use self-attention for their main blocks.

```py
class Head:
    def __init__(self, input_size, attention_size):
        self.dk = attention_size
        self.W_q = Layer(input_size, attention_size)
        self.W_k = Layer(input_size, attention_size)
        self.W_v = Layer(input_size, attention_size)

    def forward(self, X):
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        A = softmax(attention_linear(K, Q))
        output = linear(A, transpose(V))

        self.last_Q = copy_md_list(Q)
        self.last_K = copy_md_list(K)
        self.last_A = copy_md_list(A)
        self.last_V = copy_md_list(V)
        return output

    def __call__(self, X):
        return self.forward(X)
```

`linear(A, transpose(V))` is another matrix multiplication. Each row of `A` contains one token's attention weights, and each row of the result is that token's new vector. We transpose `V` because our small `linear` helper expects the rows of its second argument to be output vectors.

The backward pass is more involved than the previous layers because the output depends on `Q`, `K`, and `V` at once. The important part is still familiar: calculate the gradient for each operation, then pass gradients through the three learned layers. Since all three layers started from the same input, their input gradients must be added together.

```py
    def backward(self, upstream_gradient_batch, learning_rate):
        Q, K = self.last_Q, self.last_K
        A, V = self.last_A, self.last_V
        sequence_length = len(A)
        scale = 1.0 / sqrt(self.dk)

        dA = [
            [sum(upstream_gradient_batch[t][j] * V[s][j]
                 for j in range(self.dk))
             for s in range(sequence_length)]
            for t in range(sequence_length)
        ]
        dV = [
            [sum(A[t][s] * upstream_gradient_batch[t][j]
                 for t in range(sequence_length))
             for j in range(self.dk)]
            for s in range(sequence_length)
        ]

        dS = []
        for t in range(sequence_length):
            row_dot = sum(A[t][s] * dA[t][s]
                          for s in range(sequence_length))
            dS.append([
                A[t][s] * (dA[t][s] - row_dot)
                for s in range(sequence_length)
            ])

        dQ = [[scale * sum(dS[t][s] * K[s][j]
                           for s in range(sequence_length))
               for j in range(self.dk)]
              for t in range(sequence_length)]
        dK = [[scale * sum(dS[t][s] * Q[t][j]
                           for t in range(sequence_length))
               for j in range(self.dk)]
              for s in range(sequence_length)]

        dX_q = self.W_q.backward(dQ, learning_rate)
        dX_k = self.W_k.backward(dK, learning_rate)
        dX_v = self.W_v.backward(dV, learning_rate)
        return sum_2d(dX_q, dX_k, dX_v)
```

The softmax gradient looks scary, but it is the same general process as before: the gradient tells us how changing each score would change the final error. The mask does not need special backward code. Its forbidden scores became zero after softmax, so no gradient flows through them.

## Multi-Head Attention

One head can learn one kind of relationship. Maybe it learns to look for the subject of a sentence. Another may learn to look for nearby words, punctuation, or previous mentions of the same thing. Rather than hoping one head learns all of those at once, transformers use several heads in parallel.

```py
class MultiHead:
    def __init__(self, input_size, num_heads=4,
                 attention_size=None, output_size=None):
        if attention_size is None:
            attention_size = input_size
        if output_size is None:
            output_size = input_size
        self.attention_size = attention_size
        self.num_heads = num_heads
        self.heads = [
            Head(input_size, attention_size)
            for _ in range(num_heads)
        ]
        self.W_o = Layer(num_heads * attention_size, output_size)

    @staticmethod
    def concat(*batches):
        return [sum(rows, []) for rows in zip(*batches)]

    def forward(self, X):
        pre_out = self.concat(*[head(X) for head in self.heads])
        return self.W_o(pre_out)
```

Each head gets the same input but has different randomly initialized weights, so training can make them specialize. We join their output vectors together with `concat`, then use one final layer, `W_o`, to mix the heads back into the desired output size.

The backward pass reverses this: first backpropagate through `W_o`, split its gradient into one piece per head, backpropagate each head, then add the input gradients. That is all multi-head attention is: several copies of the same mechanism running beside each other.

At this point we have the main ingredient of a transformer. A full transformer block adds a residual connection, layer normalization, and a small feed-forward neural net around this attention layer. Those additions make deep stacks of blocks train more reliably, but the part that lets tokens share information is the attention mechanism above.

## Transformer Block

We have attention now, but an attention layer by itself is not quite a transformer. A transformer is built by stacking blocks. Each block has an attention section, a small regular neural net, residual connections, and layer normalization. The pattern is repeated over and over, which is why understanding one block gets you most of the way there.

The flow through our block looks like this:

$$
X \rightarrow \text{Multi-Head Attention} \rightarrow +X \rightarrow \text{LayerNorm} \rightarrow \text{MLP} \rightarrow +X \rightarrow \text{LayerNorm}
$$

The two additions are residual connections. Instead of making every new layer completely replace its input, we let the input take a shortcut around the layer and add it back at the end. This is useful because a layer can learn a small change to the existing representation instead of needing to recreate everything from scratch. It also gives gradients an easier route backward when many blocks are stacked.

## Layer Normalization

Layer normalization keeps each token's vector from getting too wild as it passes through many layers. For one token vector $x$, we find its mean and variance, then normalize every value:

$$
\mu = \frac{1}{d}\sum_{i=1}^{d}x_i
$$

$$
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2
$$

$$
\operatorname{LayerNorm}(x)_i = \gamma_i\frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_i
$$

`epsilon` is a tiny number that stops us from dividing by zero. `gamma` and `beta` are learned vectors. At the start, `gamma` is all ones and `beta` is all zeroes, so the layer simply normalizes. Training can later scale and shift each position if that is useful.

```py
class LayerNorm:
    def __init__(self, size, epsilon=1e-5):
        self.size = size
        self.epsilon = epsilon
        self.gamma = [1.0] * size
        self.beta = [0.0] * size

    def forward(self, X):
        self.last_normalized = []
        self.last_inverse_std = []
        output = []
        for row in X:
            mean = sum(row) / self.size
            variance = sum((value - mean) ** 2 for value in row) / self.size
            inverse_std = 1.0 / sqrt(variance + self.epsilon)
            normalized = [
                (value - mean) * inverse_std
                for value in row
            ]
            self.last_normalized.append(normalized)
            self.last_inverse_std.append(inverse_std)
            output.append([
                self.gamma[j] * value + self.beta[j]
                for j, value in enumerate(normalized)
            ])
        return output

    def __call__(self, X):
        return self.forward(X)
```

Unlike batch normalization, layer normalization works across the vector size of one token. It does not care how long the sequence is or how many sequences are in a batch. That makes it a good fit for transformers, where sequence lengths can vary.

The backward pass follows the same pattern as the other layers. It computes gradients for `gamma`, `beta`, and the original input vector, then updates `gamma` and `beta` using the average gradient across the sequence.

```py
    def backward(self, upstream_gradient_batch, learning_rate):
        gamma_gradient = [0.0] * self.size
        beta_gradient = [0.0] * self.size
        input_gradient = []

        for normalized, inverse_std, upstream in zip(
                self.last_normalized,
                self.last_inverse_std,
                upstream_gradient_batch):
            scaled = [upstream[j] * self.gamma[j]
                      for j in range(self.size)]
            mean_gradient = sum(scaled) / self.size
            normalized_gradient = sum(
                scaled[j] * normalized[j]
                for j in range(self.size)
            ) / self.size
            input_gradient.append([
                inverse_std * (
                    scaled[j] - mean_gradient
                    - normalized[j] * normalized_gradient
                )
                for j in range(self.size)
            ])
            for j in range(self.size):
                gamma_gradient[j] += upstream[j] * normalized[j]
                beta_gradient[j] += upstream[j]

        count = len(upstream_gradient_batch)
        for j in range(self.size):
            self.gamma[j] -= learning_rate * gamma_gradient[j] / count
            self.beta[j] -= learning_rate * beta_gradient[j] / count
        return input_gradient
```

## The Feed-Forward Network

Attention lets tokens exchange information. After that, each token goes through the same small neural net independently. This is usually called the feed-forward network, or MLP. The network is wider in its hidden layer, uses an activator, then returns to the original vector size. This version of `Neural_Net` stops after its last `Layer`; it does not use the final softmax from our digit classifier, because these vectors are not class probabilities.

```py
self.mlp = Neural_Net(
    attention_size,
    4 * attention_size,
    output_size
)
```

This is the same kind of neural net we built earlier. It does not mix tokens together; attention already did that. Applying the same MLP to every row gives the transformer another chance to reshape the information each token collected.

## Putting It Together

Here is the forward pass for the full block. Our code allows the input and output sizes to differ. When they do, the residual path needs a `Layer` to project the input to the right size before we add it.

```py
class TransformerBlock:
    def __init__(self, att_input_size, att_output_size,
                 mlp_output_size, num_heads=4,
                 feed_forward_size=None):
        if feed_forward_size is None:
            feed_forward_size = 4 * att_output_size

        self.attention = MultiHead(
            att_input_size,
            num_heads=num_heads,
            output_size=att_output_size
        )
        self.attention_residual = (
            Layer(att_input_size, att_output_size)
            if att_input_size != att_output_size else None
        )
        self.attention_norm = LayerNorm(att_output_size)

        self.mlp = Neural_Net(
            att_output_size,
            feed_forward_size,
            mlp_output_size
        )
        self.mlp_residual = (
            Layer(att_output_size, mlp_output_size)
            if att_output_size != mlp_output_size else None
        )
        self.mlp_norm = LayerNorm(mlp_output_size)

    def forward(self, X):
        residual = X if self.attention_residual is None \
            else self.attention_residual(X)
        attention_output = self.attention(X)
        attention_output = [
            [a + b for a, b in zip(attention_row, residual_row)]
            for attention_row, residual_row in zip(
                attention_output, residual
            )
        ]
        normalized_attention = self.attention_norm(attention_output)

        mlp_residual = normalized_attention \
            if self.mlp_residual is None \
            else self.mlp_residual(normalized_attention)
        mlp_output = self.mlp(normalized_attention)
        output = [
            [a + b for a, b in zip(mlp_row, residual_row)]
            for mlp_row, residual_row in zip(mlp_output, mlp_residual)
        ]
        return self.mlp_norm(output)

    def __call__(self, X):
        return self.forward(X)
```

The exact order here is `attention -> add -> normalize -> MLP -> add -> normalize`. You may see other transformers normalize before attention or the MLP instead. Both arrangements are used; ours is the original post-normalization style and keeps the code easy to follow.

Backpropagation is simply this flow in reverse. The gradient coming out of each addition has to travel down both paths: one copy goes through the main layer and one copy goes down the residual shortcut. When both paths reach the same earlier input, we add their gradients together.

```py
    def backward(self, upstream_gradient_batch, learning_rate):
        grad = self.mlp_norm.backward(upstream_gradient_batch, learning_rate)
        grad_residual = copy_md_list(grad)
        grad_mlp = self.mlp.backward(grad, learning_rate)
        grad_mlp_residual = (
            grad_residual if self.mlp_residual is None
            else self.mlp_residual.backward(grad_residual, learning_rate)
        )
        grad = sum_2d(grad_mlp, grad_mlp_residual)

        grad = self.attention_norm.backward(grad, learning_rate)
        grad_residual = copy_md_list(grad)
        grad_attention = self.attention.backward(grad, learning_rate)
        grad_attention_residual = (
            grad_residual if self.attention_residual is None
            else self.attention_residual.backward(
                grad_residual, learning_rate
            )
        )
        return sum_2d(grad_attention, grad_attention_residual)
```

That is a transformer block. A GPT-style model is mostly a token embedding layer, positional information, a stack of blocks like this one, and a final layer that turns the last vectors into probabilities for the next token. We have now built the part where the actual context mixing happens.
