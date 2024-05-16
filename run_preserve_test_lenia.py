import os
import preserve_tester 
import lenia
import mathutil
import torch
#100*100のグリッドで実験していた
#f=0.06, k=0.08で実験していた
explist=[]
lenialist_=os.listdir("creature_data/lenia/")
lenialist=[ {"pathname":"creature_data/lenia/"+st} for st in lenialist_ ]
rdlenialist=[ {"pathname":"creature_data/rdlenia/"+st,
               "uval":0.1,"vval":0,"f":0.06,"k":0.08} for st in lenialist_]
mutil=mathutil.MathUtil()

for i in range(len(lenialist)):
    #ある生物に関してLenia環境でニッチを調べる
    targ_lenia=lenialist[i]
    creature_lenia=mutil.load_creature_from_csv(targ_lenia["pathname"])
    print(targ_lenia["pathname"])
    a_arr=torch.zeros(1,1,150,150)
    a_arr=mutil.place_creature(creature_lenia,a_arr)
    filename=targ_lenia["pathname"].replace("creature_data/lenia/","").replace(".csv","")
    model_lenia=lenia.Lenia_Simple(a_arr,"exponential",0,0,13,10)
    tester_lenia=preserve_tester.PreserveTesterLenia(40,100,model_lenia,filename)
    tester_lenia.test()

    #ある生物に関してRDLenia環境でニッチを調べる