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

class Preserve_tester:
    def __init__(self,num_steps,phase_diagram_grid_size,model:RDLenia,pattern_file_name,uval,vval):
        self.num_steps=num_steps#40
        self.mu=MathUtil()
        self.phase_diagram_grid_size=phase_diagram_grid_size#100
        result=[[False for j in range(phase_diagram_grid_size)]
                for i in range(phase_diagram_grid_size)]
        self.result=np.array(result)
        self.delta_sigma=0.005
        self.delta_mu=0.005
        self.model=model
        self.pattern_file_name=pattern_file_name
        self.uval=uval
        self.vval=vval
    
    def test(self):
        for i in range (self.phase_diagram_grid_size):
            for j in range(self.phase_diagram_grid_size):
                array_creature=self.mu.load_creature_from_csv(self.pattern_file_name)
                h=100;w=100;
                u=torch.tensor(np.ones((1,1,h,w))).float()*self.uval
                v=torch.tensor(np.ones((1,1,h,w))).float()*self.vval
                self.model.a=torch.tensor(np.zeros(1,1,h,w)).float()
                self.model.u=u
                self.model.v=v
                self.model.a=self.mu.place_creature(array_creature,self.model.a)

                pre_a=self.mu.rdltensor2arr(self.model.a)
                pre_mass_avg=self.mu.calc_mass_avg(pre_a)
                pre_mass_std=self.mu.calc_mass_stdev(pre_a)
                alog=[]
                for steps in range(self.num_steps):
                    self.model.step()
                    alog.append(self.model.a.detach().numpy())
                aft_a=np.stack(alog)
                aft_mass_avg=self.mu.calc_mass_avg(aft_a)
                aft_mass_std=self.mu.calc_mass_stdev(aft_a)

                #可視化テスト
                #sarr=rdl.a.squeeze().numpy()
                #plt.imshow(sarr)
                #plt.show()

                #生存判定。
                b=True
                if math.fabs(pre_mass_avg-aft_mass_avg)>=0.3*pre_mass_avg:
                    b=False

                self.result[i,j]=b
            
        plt.imshow(np.flip(self.result,axis=0),
                   extent=(0,self.delta_sigma*self.phase_diagram_grid_size,
                           0,self.delta_mu*self.phase_diagram_grid_size))
        plt.title(self.pattern_file_name.replace(".csv",""))
        plt.xlabel('sigma')
        plt.ylabel('mu')
        plt.savefig('images/{}.png'.format(self.pattern_file_name.replace(".csv","")))