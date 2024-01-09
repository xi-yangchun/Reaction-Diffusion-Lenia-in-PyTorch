import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rdlenia import RDLenia
import os
from stat_lenia import Stat_Lenia

g=0.01
f=[g*i for i in range(int(0.1/g))]
k=[g*i for i in range(int(0.1/g))]
f_lenia=[g*i for i in range(int(0.1/g))]

num_steps=2000
exp_prefix='2023_1114_lenia'
try:
    os.makedirs("result/"+exp_prefix)
except:
    print("experiment exists")

import pandas as pd
# Create a dictionary with your data
metadata = {
    'exp_index': [],
    'g_mu':[],
    'g_sigma':[],
    'f': [],
    'k': [],
    'Du': [],
    'Dv': [],
    'f_lenia': [],
    'gamma':[]
}

for i in range(int(0.1/g)):
    for j in range(int(0.1/g)):
        for q in range(int(0.1/g)):
            a=torch.tensor(np.random.rand(1,1,200,200)).float()
            u=torch.tensor(np.ones((1,1,200,200))).float()
            v=torch.tensor(np.zeros((1,1,200,200))).float()
            g_mu=0.15;g_sigma=0.017;Du=2e-3;Dv=1e-3;gamma=0.1
            rdl=RDLenia(a,"exponential",g_mu,g_sigma,u,v,Du,Dv,f[i],k[q],
                gamma,f_lenia[j],13,5)
            sl=Stat_Lenia()
            for steps in range(num_steps):
                rdl.step()
                sl.update_stat(rdl)
            exp_index="{}_{}_{}".format(i,j,q)
            metadata["exp_index"].append(exp_index)
            metadata["g_mu"].append(g_mu)
            metadata["g_sigma"].append(g_sigma)
            metadata["f"].append(f[i])
            metadata["k"].append(k[q])
            metadata["Du"].append(Du)
            metadata["Dv"].append(Dv)
            metadata["f_lenia"].append(f_lenia[j])
            metadata["gamma"].append(gamma)
            df = pd.DataFrame(sl.get_stat())
            # Save the DataFrame to a CSV file
            df.to_csv('result/{}/{}.csv'.format(exp_prefix,exp_index), index=False)
            print(exp_index+" done")

# Create a DataFrame from the dictionary
df = pd.DataFrame(metadata)
# Save the DataFrame to a CSV file
df.to_csv('result/{}/metadata.csv'.format(exp_prefix), index=False)