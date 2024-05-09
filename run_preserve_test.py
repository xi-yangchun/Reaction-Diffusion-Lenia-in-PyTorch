import os
import preserve_tester 
import rdlenia
#100*100のグリッドで実験していた
#f=0.06, k=0.08で実験していた
explist=[]
lenialist_=os.listdir("creature_data/lenia/")
lenialist=[ {"pathname":"creature_data/lenia/"+st,
             "uval":0,"vval":0,"f":0,"k":0} for st in lenialist_ ]
rdlenialist_=os.listdir("creature_data/rdlenia/")
rdlenialist=[ {"pathname":"creature_data/rdlenia/"+st,
               "uval":0.1,"vval":0,"f":0.06,"k":0.08} for st in rdlenialist_]
explist=rdlenialist+lenialist

for exp in explist:
    model=rdlenia.RDLenia(None,"exokernel",0,0,None,None,2e-3,1e-3,exp["f"],exp["k"],0.1,0.05,13,10)
    pt=preserve_tester.Preserve_tester(40,100,model,exp["pathname"],exp["u"],exp["v"])
    pt.test()