import cv2
import torch
import numpy as np
import mathutil
from rdlenia import RDLenia
class Stat_Lenia:
    def __init__(self,len_queue=2):
        self.len_queue=len_queue
        self.queue_a=[]
        self.queue_u=[]
        self.queue_v=[]
        self.stat={'time':[],'square_optical_flow':[],
                   'avg_spatial_entropy':[],
                   'stdev_spatial_entropy':[]}
        self.steps=0
    def update_stat(self,lenia:RDLenia):
        self.steps+=1

        if self.steps%lenia.T==0:
            a,u,v=lenia.get_current_state()
            a=a.detach().squeeze().numpy()
            u=u.detach().squeeze().numpy()
            v=v.detach().squeeze().numpy()
            self.queue_a.append(a)
            self.queue_u.append(u)
            self.queue_v.append(v)
            
            if len(self.queue_a)<self.len_queue:
                return
            if len(self.queue_a)>self.len_queue:
                self.queue_a.pop(0)
                self.queue_u.pop(0)
                self.queue_v.pop(0)
            opf=self.calc_optical_flow(self.queue_a[0],self.queue_a[1],lenia.dx)
            spe=self.calc_spatial_entropy(self.queue_a[-1],0.2,3)
            self.stat['time'].append(lenia.dt*self.steps)
            self.stat['square_optical_flow'].append(np.sum(opf*opf))
            self.stat['avg_spatial_entropy'].append(np.mean(spe))
            self.stat['stdev_spatial_entropy'].append(np.std(spe))
        
    def get_stat(self):
        return self.stat

    def calc_optical_flow(self,arr0,arr1):
        s=cv2.calcOpticalFlowPyrLK(arr0,arr1)
        return s
    
    def calc_spatial_entropy(self,arr:np.array,resol,window_radius):
        n=int(1/resol)
        mu=mathutil.MathUtil()
        occr=[]
        for i in range(n):
            occr.append((arr>=resol*i+1e-9) & (arr<resol*(i+1)+1e-9))
        wr=window_radius
        occr_window=[]

        for k in range(n):
            occr_in_bin=[]
            for i in range(2*wr+1):
                dy=i-wr
                for j in range(2*wr+1):
                    dx=j-wr
                    occr_in_bin.append(mu.roll_np_2darr(occr[k],dx,dy))
            occr_window.append(occr_in_bin)
        
        freq=[]
        for occr_in_bin in occr_window:
            freq.append(np.sum(np.array(occr_in_bin),axis=0))

        h,w=arr.shape[0],arr.shape[1]
        spatial_entropy=np.zeros((h,w))
        for freq_in_bin in freq:
            rate_in_bin=freq_in_bin/(h*w)+1e-9
            spatial_entropy-=np.log(rate_in_bin)*rate_in_bin
        return spatial_entropy
    
    def calc_optical_flow(self,arr_old:np.array,arr_new:np.array,resol):
        #calc farneback optical flow
        flow=np.zeros((2,arr_old.shape[0],arr_old.shape[1]))
        flow=cv2.calcOpticalFlowFarneback(arr_old,arr_new,flow,0.5,3,int(1/resol)//2,3,5,1.2,0)
        #flow is represented by [Hue,Value]
        return flow