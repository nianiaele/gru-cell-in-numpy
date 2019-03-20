import torch
import torch.nn as nn
import numpy as np
import itertools

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


class GRU_Cell:
    """docstring for GRU_Cell"""
    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.hh=h

        self.Wzh = np.random.randn(h,h)
        self.Wrh = np.random.randn(h,h)
        self.Wh  = np.random.randn(h,h)

        self.Wzx = np.random.randn(h,d)
        self.Wrx = np.random.randn(h,d)
        self.Wx  = np.random.randn(h,d)



        self.dWzh = np.zeros((h,h))
        self.dWrh = np.zeros((h,h))
        self.dWh  = np.zeros((h,h))

        self.dWzx = np.zeros((h,d))
        self.dWrx = np.zeros((h,d))
        self.dWx  = np.zeros((h,d))

        self.z_act = Sigmoid()
        self.r_act = Sigmoid()
        self.h_act = Tanh()


    def forward(self, x, h):
        # input:
        # 	- x: shape(input dim),  observation at current time-step
        # 	- h: shape(hidden dim), hidden-state at previous time-step
        #
        # output:
        # 	- h_t: hidden state at current time-step
        self.x=x
        self.h=h


        self.z1=self.Wzh@self.h
        self.z2=self.Wzx@self.x
        self.z3=self.z1+self.z2
        self.zt=self.z_act(self.z3)

        self.z4=self.Wrh@h
        self.z5=self.Wrx@x
        self.z6=self.z4+self.z5
        self.rt=self.r_act(self.z6)

        self.z7=self.rt*self.h
        self.z8=self.Wh@self.z7
        self.z9=self.Wx@self.x
        self.z10=self.z8+self.z9
        self.hthat=self.h_act(self.z10)

        self.z11=1-self.zt
        self.z12=self.z11*self.h
        self.z13=self.zt*self.hthat
        self.ht=self.z12+self.z13

        return self.ht


    def backward(self, delta):
        # input:
        # 	- delta: 	shape(hidden dim), summation of derivative wrt loss from next layer at
        # 			same time-step and derivative wrt loss from same layer at
        # 			next time-step
        #
        # output:
        # 	- dx: 	Derivative of loss wrt the input x
        # 	- dh: 	Derivative of loss wrt the input hidden h

        self.init_deriv()


        #first wave
        a,b =self.deriv(delta,'+',self.z12,self.z13)
        self.dz12+=a
        self.dz13+=b
        a,b=self.deriv(self.dz13,"*",self.zt,self.hthat)
        self.dzt+=a
        self.dhthat+=b
        a,b=self.deriv(self.dz12,'*',self.z11,self.h)
        self.dz11+=a
        self.dh+=b
        a,b=self.deriv(self.dz11,'-',1,self.zt)
        self.dz11+=b

        #second wave
        a = self.deriv(self.dhthat,'act',0,0,self.h_act)
        self.dz10+=a
        a,b=self.deriv(self.dz10,'+',self.z8,self.z9)
        self.dz8+=a
        self.dz9+=b
        a,b=self.deriv(self.dz9,'@',self.Wx,self.x)
        self.dWx+=a
        self.dx+=b
        a,b=self.deriv(self.dz8,'@',self.Wh,self.z7)
        self.dWh+=a
        self.dz7+=b
        a,b=self.deriv(self.dz7,'*',self.rt,self.h)
        self.drt+=a
        self.dh+=b

        #third wave
        a=self.deriv(self.drt,'act',0,0,self.r_act)
        self.dz6+=a
        a,b=self.deriv(self.z6,'+',self.z4,self.z5)
        self.dz4+=a
        self.dz5+=b
        a,b=self.deriv(self.dz5,'@',self.Wrx,self.x)
        self.dWrx+=a
        self.dx+=b
        a,b=self.deriv(self.dz4,'@',self.Wrh,self.h)
        self.dWrh+=a
        self.dh+=b

        #fourth wave
        a=self.deriv(self.dzt,'act',0,0,self.z_act)
        self.dz3+=a
        a,b=self.deriv(self.dz3,'+',self.z1,self.z2)
        self.dz1+=a
        self.dz2+=b
        a,b=self.deriv(self.dz2,'@',self.Wzx,self.x)
        self.dWzx+=a
        self.dx+=b
        a,b=self.deriv(self.dz1,'@',self.Wzh,self.h)
        self.dWzh+=a
        self.dh+=b


        return self.dx,self.dh






    def init_deriv(self):
        self.dz13=0
        self.dz12=0
        self.dz11=0
        self.dhthat=0
        self.dz10=0
        self.dz9=0
        self.dz8=0
        self.dz7=0
        self.drt=0
        self.dz6=0
        self.dz5=0
        self.dz4=0
        self.dzt=0
        self.dz3=0
        self.dz2=0
        self.dz1=0
        self.dh=0
        self.dx=0

        h=self.hh
        d=self.d
        self.dWzh = np.zeros((h,h))
        self.dWrh = np.zeros((h,h))
        self.dWh  = np.zeros((h,h))

        self.dWzx = np.zeros((h,d))
        self.dWrx = np.zeros((h,d))
        self.dWx  = np.zeros((h,d))

    def deriv(self,dz,operator,x=None,y=None,act_fn=None):
        if operator==None:
            return dz
        elif operator=='*':
            return dz*np.transpose(x), dz*np.transpose(y)
        elif operator=='@':
            return dz@y,x@dz#correct?
        elif operator=='+':
            return dz,dz
        elif operator=='-':
            return dz,-dz
        elif operator=='act':
            assert act_fn!=None

            return dz*act_fn.backward()
        else:
            raise Exception





if __name__ == '__main__':
    test()









