import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mathutil import MathUtil

class GSCott:
    def __init__(self,u:torch.tensor,v:torch.tensor,Du,Dv,f,k,R,T):
        self.u=u;self.v=v;
        self.Du=Du;self.Dv=Dv
        self.f=f;self.k=k;
        self.R=R;self.T=T
        self.dx=1.0/R
        self.dt=1.0/T
        self.ker=torch.tensor(np.array(
            [[[[0,1,0],
             [1,-4,1],
             [0,1,0]]]]
            )).float()

    def step(self)->dict:
        u_pad=F.pad(self.u,[1,1,1,1],mode='circular')
        v_pad=F.pad(self.v,[1,1,1,1],mode='circular')
        lap_u=F.conv2d(u_pad,self.ker)/(self.dx**2)
        lap_v=F.conv2d(v_pad,self.ker)/(self.dx**2)
        self.u+=self.dt*(self.Du*lap_u-self.u*(self.v**2)+self.f*(1-self.u))
        self.v+=self.dt*(self.Dv*lap_v+self.u*(self.v**2)-(self.f+self.k)*self.v)
        return {"u":self.u,"v":self.v}