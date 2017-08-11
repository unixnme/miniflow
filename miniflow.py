import numpy as np

class Layer(object):
    """
    base class for all inputs and outputs and intermediate layers
    """
    def __init__(self, inbound_nodes=[]):
        if type(self) == Layer:
            raise Exception("Layer itself should never be instantiated")

        self.inbound_nodes = inbound_nodes
        self.outbound_nodes = []
        # add itself to output nodes
        for node in inbound_nodes:
            node.outbound_nodes.append(self)

    def forward(self):
        # nothing to do
        pass

    def backward(self):
        # set self.grad_cost
        self.grad_cost = 0
        for node in self.outbound_nodes:
            self.grad_cost += node.grad[self]
        # initialize self.grad
        self.grad = {}

class Input(Layer):
    def __init__(self):
        super(Input, self).__init__()

    def forward(self):
        # self.value set during topological_sort
        pass

    def backward(self):
        super(Input, self).backward()

class Linear(Layer):
    def __init__(self, X, W, b):
        super(Linear, self).__init__([X, W, b])
        self.X = self.inbound_nodes[0]
        self.W = self.inbound_nodes[1]
        self.b = self.inbound_nodes[2]

    def forward(self):
        self.value = np.dot(self.X.value, self.W.value) + self.b.value

    def backward(self):
        super(Linear, self).backward()
        # (N, n) = (N, m) * (m, n)
        self.grad[self.X] = np.dot(self.grad_cost, self.W.value.T)
        # (n, m) = (n, N) * (N, m)
        self.grad[self.W] = np.dot(self.X.value.T, self.grad_cost)
        # (m, ) = sum(N, m, axis=0)
        self.grad[self.b] = np.sum(self.grad_cost, axis=0)

class MSE(Layer):
    def __init__(self, pred, true):
        super(MSE, self).__init__([pred, true])
        self.pred = self.inbound_nodes[0]
        self.true = self.inbound_nodes[1]

    def forward(self):
        self.diff = self.pred.value - self.true.value
        self.value = 0.5 * np.mean(self.diff ** 2)

    def backward(self):
        super(MSE, self).backward()
        self.grad[self.pred] = self.diff / self.pred.value.shape[0]
        self.grad[self.true] = -self.grad[self.pred]

class Sigmoid(Layer):
    def __init__(self, x):
        super(Sigmoid, self).__init__([x])
        self.x = x

    def forward(self):
        self.value = 1 / (1 + np.exp(-self.x.value))

    def backward(self):
        super(Sigmoid, self).backward()
        self.grad[self.x] = self.value * (1 - self.value)


def topological_sort(feed_dict):
    """
    Sort the layers in topological order using Kahn's Algorithm.
    `feed_dict`: A dictionary where the key is a `Input` Layer and the value is the respective value feed to that Layer.
    Returns a list of sorted layers.
    """

    input_layers = [n for n in feed_dict.keys()]

    G = {}
    layers = [n for n in input_layers]
    while len(layers) > 0:
        n = layers.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            layers.append(m)

    L = []
    S = set(input_layers)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_and_backward(graph):
    """
    Performs a forward pass and a backward pass through a list of sorted Layers.
    Arguments:
        `graph`: The result of calling `topological_sort`.
    """
    # Forward pass
    for n in graph:
        n.forward()

    # Backward pass
    # see: https://docs.python.org/2.3/whatsnew/section-slices.html
    for n in graph[::-1]:
        n.backward()


def sgd_update(trainables, learning_rate=1e-2):
    """
    Updates the value of each trainable with SGD.
    Arguments:
        `trainables`: A list of `Input` Layers representing weights/biases.
        `learning_rate`: The learning rate.
    """
    # Performs SGD
    #
    # Loop over the trainables
    for t in trainables:
        # Change the trainable's value by subtracting the learning rate
        # multiplied by the partial of the cost with respect to this
        # trainable.
        partial = t.grad_cost
        t.value -= learning_rate * partial