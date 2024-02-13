import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rdlenia import RDLenia
import os
from stat_lenia import Stat_Lenia
from mathutil import MathUtil
import math
from matplotlib import pyplot as plt

num_steps=40
mu=MathUtil()
phase_grid_size=100
result=[[False for j in range(phase_grid_size)] for i in range(phase_grid_size)]
result=np.array(result)
dsig=0.005
dmu=0.005
name='rdlenia_A0'

for i in range (phase_grid_size):
    for j in range(phase_grid_size):
        f=0.06
        k=0.08
        Du=2e-3;Dv=1e-3;
        f_lenia=0.05
        g_mu=dmu*i;g_sigma=dsig*j;gamma=0.1

        a=torch.tensor(np.zeros((1,1,100,100))).float()
        u=torch.tensor(np.ones((1,1,100,100))).float()
        v=torch.tensor(np.zeros((1,1,100,100))).float()
        array_orbium=mu.load_creature_from_csv("creature_data/lenia/rdlenia_A0.csv")

        rdl=RDLenia(a,"exponential",g_mu,g_sigma,u,v,Du,Dv,f,k,
            gamma,f_lenia,13,5)
        rdl.a=mu.place_creature(array_orbium,rdl.a)

        pre_a=mu.rdltensor2arr(rdl.a)
        pre_mass_avg=mu.calc_mass_avg(pre_a)
        pre_mass_std=mu.calc_mass_stdev(pre_a)
        alog=[]
        for steps in range(num_steps):
            rdl.step()
            alog.append(rdl.a.detach().numpy())
        aft_a=np.stack(alog)
        aft_mass_avg=mu.calc_mass_avg(aft_a)
        aft_mass_std=mu.calc_mass_stdev(aft_a)

        #可視化テスト
        #sarr=rdl.a.squeeze().numpy()
        #plt.imshow(sarr)
        #plt.show()

        #生存判定。
        b=True
        if math.fabs(pre_mass_avg-aft_mass_avg)>=0.3*pre_mass_avg:
            b=False

        result[i,j]=b
        print((i,j))
    
plt.imshow(np.flip(result,axis=0),extent=(0,dsig*phase_grid_size,0,dmu*phase_grid_size))
plt.title(name)
plt.xlabel('sigma')
plt.ylabel('mu')
plt.savefig('images/{}.png'.format(name))