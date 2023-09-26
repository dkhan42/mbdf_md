import numpy as np
from qml.math import svd_solve
from scipy.linalg import cho_solve
from qml.kernels import get_atomic_local_kernel, get_atomic_local_gradient_kernel
from qml.kernels import get_local_gradient_kernel, get_local_kernel

def KRR_oqml(xt, dxt, qt, et, ft, xte, dxte, qte, sigma, lam):
    """
    energy and forces using oqml
    """
    kte = get_atomic_local_kernel(xt, xt, qt, qt, sigma)
    kt = get_atomic_local_gradient_kernel(xt, xt, dxt, qt, qt, sigma)
    c = np.concatenate((kte, kt))
    y = np.concatenate((et, ft.flatten()))
    y = y.astype(float)
    alpha = svd_solve(c, y, rcond=lam)
    Ks_force  = get_atomic_local_gradient_kernel(xt,xte,dxte,qt,qte,sigma)
    Ks_energy = get_atomic_local_kernel(xt,xte,qt,qte,sigma)
    eYt = np.dot(Ks_energy, alpha)
    fYt = np.dot(Ks_force, alpha)
    return eYt, fYt

def krr_local(rtrain, drtrain, qt, ftr, rtest, drtest, qtest, sigma, lam):
    ktr_local_grad = get_local_gradient_kernel(rtrain, rtrain, drtrain, 
                                                   qt, qt, sigma)
    alpha_local = svd_solve(ktr_local_grad, ftr.flatten().astype(float), rcond=lam)
    kte_local_grad = get_local_gradient_kernel(rtrain, rtest, drtest, 
                                           qt, qtest, sigma)
    return np.dot(kte_local_grad, alpha_local)

def krr_forces_atomic(rtrain, drtrain, qt, ftr, rtest, drtest, qtest, sigma, lam):
    """
    forces only using oqml (no Hessian block of the GPR matrix)
    """
    ktr_local_grad = get_atomic_local_gradient_kernel(rtrain, rtrain, drtrain, 
                                                   qt, qt, sigma)
    alpha_local = svd_solve(ktr_local_grad, ftr.flatten().astype(float), rcond=lam)
    kte_local_grad = get_atomic_local_gradient_kernel(rtrain, rtest, drtest, 
                                           qt, qtest, sigma)
    return np.dot(kte_local_grad, alpha_local)
