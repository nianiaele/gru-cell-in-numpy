import numpy as np
import torch
from torch.autograd import Variable


class Sigmoid:
    """docstring for Sigmoid"""
    def __init__(self):
        pass
    def forward(self, x):
        self.res = 1/(1+np.exp(-x))
        return self.res
    def backward(self):
        return self.res * (1-self.res)
    def __call__(self, x):
        return self.forward(x)


class Tanh:
    def __init__(self):
        pass
    def forward(self, x):
        self.res = np.tanh(x)
        return self.res
    def backward(self):
        return 1 - (self.res**2)
    def __call__(self, x):
        return self.forward(x)


# z_act = torch.sigmoid()
# r_act = torch.sigmoid()
# h_act = Tanh()

x=torch.from_numpy(np.array([ 1.62434536, -0.61175641, -0.52817175, -1.07296862,  0.86540763])).float()
h=torch.from_numpy(np.array([ 0.30017032, -0.35224985])).float()


x=x.view((5,1))
h=h.view((2,1))

x=Variable(x.data,requires_grad=True)
h=Variable(h.data,requires_grad=True)


h1=2
d1=5

Wzh = torch.ones([h1, h1],requires_grad=True).float()

Wrh = torch.ones([h1, h1],requires_grad=True).float()
Wh = torch.ones([h1, h1],requires_grad=True).float()

Wzx = torch.ones([h1, d1],requires_grad=True).float()
Wrx = torch.ones([h1, d1],requires_grad=True).float()
Wx = torch.ones([h1, d1],requires_grad=True).float()


z1 = Wzh.mm(h)
z2 =Wzx.mm(x)
z3 = z1 + z2
zt = torch.sigmoid(z3)

z4 = Wrh.mm(h)
z5 = Wrx.mm(x)
z6 = z4 + z5
rt = torch.sigmoid(z6)

z7 = rt * h
z8 = Wh.mm(z7)
z9 = Wx.mm(x)
z10 = z8 + z9
hthat = torch.tanh(z10)

z11 = 1 - zt
z12 = z11 * h
z13 = zt * hthat
ht = z12 +z13


delta=torch.from_numpy(np.array([[ 0.52057634], [-1.14434139]])).float()
ht.backward(delta)



print(x.grad)