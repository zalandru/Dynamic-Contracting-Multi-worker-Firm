#import Contract_approximation_check as cac
from cont_ap_ch.primitives import Parameters
p = Parameters()

from BasicContract import BasicContract
from BasicContract_r import BasicContract_r

#Starting the continunous contract
from ContinuousContract import ContinuousContract
cc=ContinuousContract(p)
(cc_J,cc_W)=cc.J()

bc=BasicContract(10,cc.js,p)

bc_opt=BasicContract_r(3,cc.js,p)
(best_r,J)=bc.J_K()

print("best_r",best_r)
print("J",J)
