import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lenia import Lenia_Simple
import os
from stat_lenia import Stat_Lenia
from mathutil import MathUtil
import math
from matplotlib import pyplot as plt

class PreserveTesterLenia:
    def __init__(self,num_steps,phase_diagram_grid_size,model:Lenia_Simple,pattern_file_name):
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
    
    def test(self):
        for i in range (self.phase_diagram_grid_size):
            for j in range(self.phase_diagram_grid_size):
                pre_a=self.mu.rdltensor2arr(self.model.a)
                pre_mass_avg=self.mu.calc_mass_avg(pre_a)
                pre_mass_std=self.mu.calc_mass_stdev(pre_a)
                self.model.mu=i*self.delta_mu
                self.model.sigma=j*self.delta_sigma
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
        plt.savefig('images/lenia_{}.png'.format(self.pattern_file_name.replace(".csv","")))

class PreserveTesterRDLenia:
    def __init__(self,num_steps,phase_diagram_grid_size,model:Lenia_Simple,pattern_file_name):
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
    
    def test(self):
        for i in range (self.phase_diagram_grid_size):
            for j in range(self.phase_diagram_grid_size):
                pre_a=self.mu.rdltensor2arr(self.model.a)
                pre_mass_avg=self.mu.calc_mass_avg(pre_a)
                pre_mass_std=self.mu.calc_mass_stdev(pre_a)
                self.model.mu=i*self.delta_mu
                self.model.sigma=j*self.delta_sigma
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
        plt.savefig('images/rdlenia_{}.png'.format(self.pattern_file_name.replace(".csv","")))