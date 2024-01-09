import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import math
import seaborn as sns

exp_prefix='2023_1114_lenia'
metadata=pd.read_csv('result/{}/metadata.csv'.format(exp_prefix))
list_of_csv=os.listdir('result/{}'.format(exp_prefix))
list_of_csv.remove('metadata.csv')
scale_of_data=10
targ_column='square_optical_flow_a'

max_f=0.1
max_k=0.1
max_f_lenia=0.1

phase_arr=np.zeros((scale_of_data,scale_of_data))

def avg_of_1exp(df:pd.DataFrame):
    return df.mean()

def paramval2int(value,maxval,scale_of_data):
    unitval=maxval/scale_of_data
    ret=0
    while(1):
        if value<10e-6:
            break
        else:
            ret+=1
            value-=unitval
    return ret

for i in range(len(list_of_csv)):
    df=pd.read_csv('result/{}/{}'.format(exp_prefix,list_of_csv[i]))
    exp_index=list_of_csv[i].replace('.csv','')
    param=metadata.query('exp_index == @exp_index')
    idx_f=paramval2int(param['f'].iloc[-1],max_f,scale_of_data)
    idx_k=paramval2int(param['k'].iloc[-1],max_k,scale_of_data)
    avg_of_res=avg_of_1exp(df)
    phase_arr[idx_k,idx_f]+=avg_of_res[targ_column]

plt.imshow(phase_arr)
plt.xlabel('f*{}'.format(max_f/scale_of_data))
plt.ylabel('k*{}'.format(max_k/scale_of_data))
plt.title('{}(targ_column: {})'.format(exp_prefix,targ_column))
plt.savefig('{}_{}.png'.format(exp_prefix,targ_column))