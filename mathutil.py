import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

class MathUtil:
    def __init__(self):
        pass
    def make_ker_array(self,kc_option:str,beta:list,R:int,dx:float):
        B=len(beta)
        arr=[[0 for i in range(2*R+1)] for j in range(2*R+1)]
        kc_func_dic={"exponential":self.kc_exponential}
        s=0
        for i in range(2*R+1):
            for j in range(2*R+1):
                ux=(R-j)*dx;uy=(R-i)*dx
                r=(ux**2+uy**2)**(0.5)
                if r<1.0:
                    arr[i][j]=beta[int((B*r)//1)]*\
                    kc_func_dic[kc_option]((B*r)%1)
                    s+=arr[i][j]
        ret=torch.tensor([[arr]]);ret=(ret/s).float()
        return ret
    def kc_exponential(self,r):
        alpha=4
        return math.exp(alpha-alpha/(4*r*(1-r)+0.00001))
    def growth_func(self,u,mu,sigma):
        return 2*torch.exp(-(u-mu)**2/(2*sigma**2))-1
    def roll_np_2darr(self,arr:np.array,dx:int,dy:int)->np.array:
        shape=arr.shape
        h=shape[0];w=shape[1]
        arr0=np.roll(arr,dx,axis=1)
        arr1=np.roll(arr0,dy,axis=0)
        return arr1
    
    def draw_filled_circle(self, array, radius ,center_x, center_y):
        height, width = array.shape

        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                if x**2 + y**2 <= radius**2:
                    new_x = center_x + x
                    new_y = center_y + y

                    if 0 <= new_x < width and 0 <= new_y < height:
                        array[new_y, new_x] = 1
        return array
    
    def fill_spots(self,array:np.array,num_spots):
        height,width=array.shape[0],array.shape[1]
        for i in range(num_spots):
            radius=random.randint(5,20)
            cx=random.randint(0,width)
            cy=random.randint(0,height)
            array=self.draw_filled_circle(array,radius,cx,cy)
        return array