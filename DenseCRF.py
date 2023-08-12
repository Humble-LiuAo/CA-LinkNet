import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
import numpy as np

def densecrf(img, probs, iter=5, NLABELS=2):
    C, H, W = probs.shape
    U = unary_from_softmax(probs)  # note: num classes is first dim
    d = dcrf.DenseCRF2D(W, H, NLABELS)
    d.setUnaryEnergy(U)
    # pairwise_energy = create_pairwise_bilateral(sdims=(10,10), schan=(0.01,), img=img, chdim=2)
    pairwise_energy = create_pairwise_bilateral(sdims=(80,80), schan=(13, 13, 13), img=img, chdim=2)
    d.addPairwiseEnergy(pairwise_energy, compat=10)
    Q = d.inference(iter)
    output = np.array(Q).reshape((C, H, W))
    return output