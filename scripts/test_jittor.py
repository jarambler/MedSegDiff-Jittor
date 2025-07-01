import jittor as jt
from jittor import Module
from jittor import nn
import numpy as np
batch_size = 100
n = 1000

class Model(Module):
    def __init__(self):
        self.layer1 = nn.Linear(1, 10)
        self.relu = nn.Relu() 
        self.layer2 = nn.Linear(10, 1)
    def execute (self,x) :
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

def get_data(n): # generate random data for training test.
    for i in range(n):
        x = np.random.rand(batch_size, 1)
        y = x*x
        yield jt.float32(x), jt.float32(y)

model = Model()
learning_rate = 0.1
optim = nn.SGD(model.parameters(), learning_rate)

for i,(x,y) in enumerate(get_data(n)):
    pred_y = model(x)
    loss = ((pred_y - y)**2)
    loss_mean = loss.mean()
    optim.step (loss_mean)
    print(f"step {i}, loss = {loss_mean.data.sum()}")
