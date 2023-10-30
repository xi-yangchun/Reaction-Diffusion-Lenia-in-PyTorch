import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mathutil import MathUtil

class RDLenia:
    def __init__(self,a,ker_option,g_mu,g_sigma,
                 u,v,Du,Dv,f,k,gamma,f_lenia,
                 R,T):
        #Common settings of field
        self.R=R;self.T=T
        self.dx=1.0/R
        self.dt=1.0/T

        #Lenia preferences
        self.a=a;a=a.detach()
        self.mu=g_mu;self.sigma=g_sigma
        self.ker=MathUtil().make_ker_array(ker_option,[1.0],self.R,self.dx)
        self.growth_func=MathUtil().growth_func

        #RDSystem(GScott) preferences
        self.u=u;self.v=v;
        u=u.detach();v=v.detach()
        self.Du=Du;self.Dv=Dv
        self.f=f;self.k=k;
        self.lap=torch.tensor(np.array(
            [[[[0,1,0],
             [1,-4,1],
             [0,1,0]]]]
            )).float()
        
        #settings of interaction betwn Lenia and RDSys
        self.gamma=gamma
        self.f_lenia=f_lenia
    
    def growth_by_chemical(self,U,mu,sigma,u,gamma):
        new_sigma=sigma+gamma*u
        return MathUtil().growth_func(U,mu,new_sigma)
    
    def step(self):
        #copy the old state of field
        a_=self.a.detach()
        u_=self.u.detach()
        v_=self.v.detach()

        #UPDATE LENIA
        r_ker=self.ker.shape[2]//2
        a_pad=F.pad(self.a,[r_ker,r_ker,r_ker,r_ker],mode='circular')
        U=F.conv2d(a_pad,self.ker)
        #growth func is affected by chiemical u(=u_)
        G=self.growth_by_chemical(U,self.mu,self.sigma,u_,self.gamma)
        self.a=self.a+self.dt*G
        self.a=torch.clip(self.a,0.0,1.0)

        #UPDATE RDSYS
        u_pad=F.pad(self.u,[1,1,1,1],mode='circular')
        v_pad=F.pad(self.v,[1,1,1,1],mode='circular')
        lap_u=F.conv2d(u_pad,self.lap)/(self.dx**2)
        lap_v=F.conv2d(v_pad,self.lap)/(self.dx**2)
        self.u+=self.dt*(self.Du*lap_u-self.u*(self.v**2)+self.f*(1-self.u))
        #Lenia state A(=a_) emitts chemical v. scale is f_lenia
        self.v+=self.dt*(self.Dv*lap_v
                         +self.u*(self.v**2)-(self.f+self.k)*self.v
                         +self.f_lenia*a_)
        #clipping chems. because Lenia breaks the chemical balance
        self.u=torch.clip(self.u,0.0,1.0)
        self.v=torch.clip(self.v,0.0,1.0)
    
    def get_current_state(self):
        return (self.a,self.u,self.v)