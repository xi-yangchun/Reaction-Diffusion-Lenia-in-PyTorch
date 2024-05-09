from matplotlib import pyplot as plt
from mathutil import MathUtil
import numpy as np
import json
with open("exokernel_creature.json","r") as f:
    src=json.load(f)
for creature in src["creatures"]:
    if "cells" in creature:
        st=creature["cells"]
        name=creature["name"]
        arr=MathUtil().rle2arr(st)
        np.savetxt("creature_data/lenia/"+name+".csv",arr,delimiter=',')
#plt.imshow(arr)
#plt.show()