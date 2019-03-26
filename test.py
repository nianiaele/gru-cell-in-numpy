from gru import GRU_Cell
import numpy as np

g=GRU_Cell(100,150)

x=np.random.randn(100)
hidden=np.random.randn(150)

output=g.forward(x,hidden)

delta=np.array(output)
dx,dh=g.backward(delta)
print(dx)
