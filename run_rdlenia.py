import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rdlenia import RDLenia
from monitor import Monitor
from multiprocessing import Process

#a=torch.tensor(np.random.rand(1,1,200,200)).float()
a=torch.tensor(np.random.rand(1,1,200,200)).float()
u=torch.tensor(np.ones((1,1,200,200))).float()
v=torch.tensor(np.zeros((1,1,200,200))).float()
rdl=RDLenia(a,"exponential",0.15,0.017,u,v,2e-5,1e-5,0.012,0.051,
           0.1,1.0,26,10)
m=Monitor()
m.run_single_channel(rdl)