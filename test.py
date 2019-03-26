from gru import GRU_Cell
import numpy as np

g=GRU_Cell(5,2)

x=np.random.randn(5)
hidden=np.random.randn(2)

output=g.forward(x,hidden)

delta=np.array([2,1]).reshape((1,2))
dx,dh=g.backward(delta)
print(dx.shape)
