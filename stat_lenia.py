import cv2
import torch
import numpy as np
import mathutil
class Stat_Lenia:
    def __init__(self,len_queue=2):
        self.len_queue=len_queue
        self.queue_a=[]
        self.queue_u=[]
        self.queue_v=[]
        self.stat=[]
    def update_stat(self,lenia):
        a,u,v=lenia.get_current_state()
        a=a.detach().squeeze().numpy()
        u=u.detach().squeeze().numpy()
        v=v.detach().squeeze().numpy()

        self.queue_a.append(a)
        self.queue_u.append(u)
        self.queue_v.append(v)
        if len(self.queue_a)<self.len_queue:
            return
        record=[]
        opf=self.calc_optical_flow()
        record.append(opf)
        

        
    def get_stat(self):
        return self.stat

    def calc_optical_flow(self,arr0,arr1):
        s=cv2.calcOpticalFlowPyrLK(arr0,arr1)
        return s
    
    def calc_spatial_entropy(self,arr:np.array,resol,window_radius):
        n=int(1/resol)
        
        mu=mathutil.MathUtil()
        for i in range(n):
            occr[i]=np.sum((arr>=resol*i)&(arr<resol*(i+1)))
        
    
    