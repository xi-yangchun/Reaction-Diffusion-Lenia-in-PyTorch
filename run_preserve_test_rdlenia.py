import os
import preserve_tester 
import rdlenia
import mathutil
import torch
#100*100のグリッドで実験していた
#f=0.06, k=0.08で実験していた
explist=[]
lenialist_=os.listdir("creature_data/lenia/")
rdlenialist=[ {"pathname":"creature_data/lenia/"+st,
               "uval":0.1,"vval":0,"f":0.06,"k":0.08} for st in lenialist_]
mutil=mathutil.MathUtil()

for i in range(len(rdlenialist)):
    #ある生物に関してLenia環境でニッチを調べる
    targ_rdlenia=rdlenialist[i]
    creature_lenia=mutil.load_creature_from_csv(targ_rdlenia["pathname"])
    a_arr=torch.zeros(1,1,150,150)
    a_arr=mutil.place_creature(creature_lenia,a_arr)
    u_arr=torch.ones(1,1,150,150)*0.5
    v_arr=torch.zeros(1,1,150,150)
    filename=targ_rdlenia["pathname"].replace("creature_data/lenia/","").replace(".csv","")
    model_lenia=rdlenia.RDLenia(a_arr,"exponential",0.1,0.1,u_arr,v_arr,2e-3,1e-3,0.06,0.08,
                                0.1,0.05,13,10)
    tester_lenia=preserve_tester.PreserveTesterRDLenia(40,100,model_lenia,filename)
    tester_lenia.test()

    #ある生物に関してRDLenia環境でニッチを調べる