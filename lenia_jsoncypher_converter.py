from matplotlib import pyplot as plt
from mathutil import MathUtil
import numpy as np
st="23.OTVM.qF$18.qHK.pCpDpLpIpApJpRXqN$18.qHsGpKqGqMX4.pLsO$18.OtUsLqLqHA5.RwN$19.uEvVrJRO6.LrS$17.KpHrFwGsV.pBpS6.uC$13.AIVpDpIpPpMsXuLqIqCsHqQ6.rU$12.GpBpKpAUNE2.tCuArUtOuApO5.rI$6.pV4.GpBpAE6.rWvGtUtDwBvOG4.pQpG$6.sDrJ2.IpEpA9.vRwOtWvBxVvF4.pEpR$4.AGrFuEsKWpKpH10.uJyHvEtKwSyOtK3.pFpQ$4.DqDqOuLvPpWpMI10.GyOwUsFtFxKxErPWVpTpI$3.QpIqQqAsDwDtSN12.vNyOtWsTuVwNtXqWqBqFR$2.VpKpAUWpAsEtWG12.pNxLwGuCuFuWtVrTqTpUC$.CpKN2.TEpJuCsV13.tIvVuWtXtLsPrNqLV$.OpC3.pDrCpXsPvCsL12.qLtOtTtAsErLqNpIC$VUT4.rVsHsJwEwOX11.pQrOsCrPqVqFpGH$pLpBQ4.sGuRtNvFyOyD10.TpXqOqRqGpMTE$.uCU4.PvHwFuTwDyOvE7.BpApMpRpLpBOF$.qDpQ5.uPyKuSsPwByJrH5.RpIpANA$2.wF5.qUyOwAsGtKwMuXpQ2.LpEpAC$2.qQA5.xFxOtLtCuUvCsEqEpSpRW$3.uE5.tBxBvBtOtUtVsLrCqFV$3.MrG4.qMuRuVtStBsMrNqLpD$4.pXqKE.CpXsLtJsPrTrBqFpDB$5.pHqKpWpSqLrHrOrDqKpPTB$6.DpKpVqFqGqApMXJ$9.EHGC!"
name="creature_data/lenia/synorbium_solidus.csv"
arr=MathUtil().rle2arr(st)
np.savetxt(name,arr,delimiter=',')
plt.imshow(arr)
plt.show()