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
    
    def calc_mass_avg(self,array:np.array):
        return np.mean(array)
    
    def calc_mass_stdev(self,array:np.array):
        return np.std(array)
    
    def rdltensor2arr(self,tsr:torch.tensor):
        return tsr.squeeze().numpy()
    
    def place_creature(self,carr:np.array,tarr:torch.tensor):
        ch=carr.shape[0]
        cw=carr.shape[1]
        th=tarr.shape[2]
        tw=tarr.shape[3]
        carr_=torch.tensor(carr)
        tarr[0,0,th//2-ch//2:th//2+ch//2,tw//2-cw//2:tw//2+cw//2]=carr_[0:ch,0:cw]
        return tarr
    
    def load_creature_from_csv(self,csv):
        return np.loadtxt(csv,delimiter=',')
    
    def rle2arr(self,st):
        DIM_DELIM={0:'', 1:'$', 2:'%', 3:'#', 4:'@A', 5:'@B', 6:'@C', 7:'@D', 8:'@E', 9:'@F'}
        DIM=2
        stacks = [[] for dim in range(DIM)]
        last, count = '', ''
        delims = list(DIM_DELIM.values())
        st = st.rstrip('!') + DIM_DELIM[DIM-1]
        for ch in st:
            if ch.isdigit(): count += ch
            elif ch in 'pqrstuvwxy@': last = ch
            else:
                if last+ch not in delims:
                    self._append_stack(stacks[0], self.ch2val(last+ch)/255, count, is_repeat=True)
                else:
                    dim = delims.index(last+ch)
                    for d in range(dim):
                        self._append_stack(stacks[d+1], stacks[d], count, is_repeat=False)
                        stacks[d] = []
                    #print('{0}[{1}] {2}'.format(last+ch, count, [np.asarray(s).shape for s in stacks]))
                last, count = '', ''
        A = stacks[DIM-1]
        max_lens = [0 for dim in range(DIM)]
        self._recur_get_max_lens(0, A, max_lens,DIM)
        self._recur_cubify(0, A, max_lens,DIM)
        return np.asarray(A)
    
    def _append_stack(self,list1, list2, count, is_repeat=False):
        list1.append(list2)
        if count != '':
            repeated = list2 if is_repeat else []
            list1.extend([repeated] * (int(count)-1))

    def ch2val(self,c):
        if c in '.b': return 0
        elif c == 'o': return 255
        elif len(c) == 1: return ord(c)-ord('A')+1
        else: return (ord(c[0])-ord('p')) * 24 + (ord(c[1])-ord('A')+25)
    
    def _recur_get_max_lens(self,dim, list1, max_lens,mdim):
        max_lens[dim] = max(max_lens[dim], len(list1))
        if dim < mdim-1:
            for list2 in list1:
                self._recur_get_max_lens(dim+1, list2, max_lens,mdim)
    
    def _recur_cubify(self,dim, list1, max_lens,mdim):
        more = max_lens[dim] - len(list1)
        if dim < mdim-1:
            list1.extend([[]] * more)
            for list2 in list1:
                self._recur_cubify(dim+1, list2, max_lens,mdim)
        else:
            list1.extend([0] * more)
    
