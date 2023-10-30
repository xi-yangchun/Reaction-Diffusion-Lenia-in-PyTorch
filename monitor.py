import pygame
import numpy
import random
import torch
from pygame.locals import*
import sys
import os
import numpy as np
import rdlenia
import stat_lenia

class Monitor:
    def __init__(self):
        pygame.init()                                             # Pygameの初期化
        self.screen = pygame.display.set_mode((600,600))
        self.clock = pygame.time.Clock() # クロックの設定。異なるPCで異なる速さの動作になることを防ぐ
        pygame.display.set_caption("viewer")                        # タイトルバーに表示する文字
        self.visarray=None
    
    def send_4dtensor_2_screen(self,lenia_tensor:torch.tensor,u_tensor:torch.tensor):
        lenia_array=(lenia_tensor.numpy()[0,0,:,:]*255).astype("int")
        u_array=(u_tensor.numpy()[0,0,:,:]*255).astype("int")
        pixel_size=3
        lenia_array=lenia_array.repeat(pixel_size,axis=0).repeat(pixel_size,axis=1)
        u_array=u_array.repeat(pixel_size,axis=0).repeat(pixel_size,axis=1)
        h=lenia_array.shape[0]
        w=lenia_array.shape[1]
        screen_arr=pygame.surfarray.pixels3d(self.screen)
        screen_arr[0:w,0:h,0]=lenia_array.T[0:w,0:h]
        screen_arr[0:w,0:h,2]=u_array.T[0:w,0:h]
        del screen_arr
    
    #test spatial entropy calculation
    def check_sumof_spatial_entropy(self,sl:stat_lenia.Stat_Lenia,
                              rdl:rdlenia.RDLenia):
        print(np.sum(sl.calc_spatial_entropy(rdl.a.numpy(),0.2,3)))

    def run_single_channel(self,rdl:rdlenia.RDLenia):
        sl=stat_lenia.Stat_Lenia()
        while (1):
            #pygame.surfarray.blit_array(self.screen,np.ones((600,600,3))*255)
            #self.update_screen_1_channel(torch.tensor([[[[0,0,0],[0,0,0],[0,0,0]]]]))
            #arr_old=rdl.a.numpy()[0,0,:,:]
            rdl.step()
            #arr_new=rdl.a.numpy()[0,0,:,:]
            #of=sl.calc_optical_flow(arr_old,arr_new,rdl.dx)
            #print(np.sum(of*of))
            self.send_4dtensor_2_screen(rdl.a,rdl.u)
            #self.check_sumof_spatial_entropy(sl,rdl)
            pygame.display.update()     # 画面を更新
            self.clock.tick(40)
            for event in pygame.event.get():
                if event.type == QUIT:  # 閉じるボタンが押されたら終了
                    pygame.quit()       # Pygameの終了(画面閉じられる)
                    sys.exit()

#m=Monitor()
#m.run_single_channel()