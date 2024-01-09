import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rdlenia import RDLenia
from monitor import Monitor
from multiprocessing import Process
from mathutil import MathUtil

a=torch.tensor(np.random.rand(1,1,200,200)).float()
#a=torch.tensor(np.zeros((1,1,200,200))).float()
#v=torch.tensor(np.zeros((1,1,200,200))).float()
v=torch.tensor(MathUtil().fill_spots(np.zeros((200,200)),30))\
.float().unsqueeze(0).unsqueeze(0)
u=torch.tensor(np.ones((1,1,200,200))).float()
rdl=RDLenia(a,"exponential",0.15,0.017,u,v,2e-3,1e-3,0.22,0.051,
           0.1,0.25,13,5)
m=Monitor()
m.run_single_channel(rdl)