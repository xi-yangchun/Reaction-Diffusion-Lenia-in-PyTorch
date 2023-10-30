import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rdlenia import RDLenia
import os
from stat_lenia import Stat_Lenia

g=0.2
f=[g*i for i in range(int(1/g))]
k=[g*i for i in range(int(1/g))]
Du=[0.00001*i for i in range(int(1/g))]
Dv=[0.00001*i for i in range(int(1/g))]
f_lenia=[g*i for i in range(int(1/g))]
gamma=[g*i for i in range(int(1/g))]

num_steps=20
exp_prefix='2023_1031_testrun_5'
try:
    os.makedirs("result/"+exp_prefix)
except:
    print("experiment exists")

import pandas as pd
# Create a dictionary with your data
metadata = {
    'exp_index': [],
    'f_lenia': [],
    'f': []
}

for i in range(int(1/g)):
    for j in range(int(1/g)):
        a=torch.tensor(np.random.rand(1,1,200,200)).float()
        u=torch.tensor(np.ones((1,1,200,200))).float()
        v=torch.tensor(np.zeros((1,1,200,200))).float()
        rdl=RDLenia(a,"exponential",0.15,0.017,u,v,2e-5,1e-5,f[i],0.051,
            0.1,f_lenia[j],26,10)
        sl=Stat_Lenia()
        for k in range(num_steps):
            rdl.step()
            sl.update_stat(rdl)
        exp_index="{}_{}".format(i,j)
        metadata["exp_index"].append(exp_index)
        metadata["f"].append(f[i])
        metadata["f_lenia"].append(f_lenia[j])
        df = pd.DataFrame(sl.get_stat())
        # Save the DataFrame to a CSV file
        df.to_csv('result/{}/{}.csv'.format(exp_prefix,exp_index), index=False)

# Create a DataFrame from the dictionary
df = pd.DataFrame(metadata)
# Save the DataFrame to a CSV file
df.to_csv('result/{}/metadata.csv'.format(exp_prefix), index=False)