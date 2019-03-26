from gru import GRU_Cell
import numpy as np

g=GRU_Cell(5,2)

x=np.array([ 1.62434536, -0.61175641, -0.52817175, -1.07296862,  0.86540763])
hidden=np.array([ 0.30017032, -0.35224985])

output=g.forward(x,hidden)



delta=np.array([[ 0.52057634, -1.14434139]])
dx,dh=g.backward(delta)

print(output)
print(g.dx)

