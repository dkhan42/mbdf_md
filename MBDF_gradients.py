import numpy as np
import numba
from copy import deepcopy

#root2,ipi=2**0.5
half_rootpi=(np.pi**0.5)/2
a2b = 1.88973

@numba.jit(nopython=True)
def erfunc(z):
    t = 1.0 / (1.0 + 0.5 * np.abs(z))
    ans = 1 - t * np.exp( -z*z -  1.26551223 +
                        t * ( 1.00002368 +
                        t * ( 0.37409196 + 
                        t * ( 0.09678418 + 
                        t * (-0.18628806 + 
                        t * ( 0.27886807 + 
                        t * (-1.13520398 + 
                        t * ( 1.48851587 + 
                        t * (-0.82215223 + 
                        t * ( 0.17087277))))))))))
    return ans


@numba.jit(nopython=True)
def hermite_polynomial(x, degree, a=1):
    if degree == 0:
        return 1
    elif degree == 1:
        return -2*a*x
    elif degree == 2:
        x1 = (a*x)**2
        return 4*x1 - 2*a
    elif degree == 3:
        x1 = (a*x)**3
        return -8*x1 - 12*a*x
    elif degree == 4:
        x1 = (a*x)**4
        x2 = (a*x)**2
        return 16*x1 - 48*x2 + 12*a**2


@numba.jit(nopython=True)
def grad_hermite(x, degree, grad, a=1):
    if degree==0:
        return np.array([0.0]*3)
    if degree==1:
        return -2*a*grad
    elif degree==2:
        return (-8*(a**2)*x)*grad
    elif degree==3:
        x1 = 2*(a**2)*(x**2)
        return 12*a*(x1+1)*grad
    elif degree==4:
        x1 = 3 - (2*(a**2)*(x**2))
        return 32*(a**2)*x1*x*grad
    

@numba.jit(nopython=True)
def gradRij(Ri, Rj, Rij):
    return (Ri - Rj)/Rij


@numba.jit(nopython=True)
def gradRji(Rj, Ri, Rij):
    return (Rj - Ri)/Rij


@numba.jit(nopython=True)
def gradCosAngj(Ri, Rj, Rk, Rij, Rik, Rjk, gRij, gRjk, angle, indices='jik'):
    
    if indices=='jik':
        grad = gRij

        term1 = (angle*grad)/Rij
        term2 = (Rk - Ri)/(Rij*Rik)

        return term2 - term1

    elif indices=='kji':
        grad1 = gRjk
        grad2 = gRij

        term1 = (angle*(-(Rij*grad1) - (Rjk*grad2)))/(Rjk*Rij)
        term2 = (2*Rj - Ri - Rk)/(Rjk*Rij)

        return term1+term2
    
    elif indices=='ikj':
        grad = gRjk

        term1 = (angle*grad)/Rjk
        term2 = (Ri - Rk)/(Rik*Rjk)

        return term2 - term1


@numba.jit(nopython=True)
def gradCosAngi(Ri, Rj, Rk, Rij, Rik, Rjk, gRij, gRik, angle, indices='jik'): #checked
    
    if indices=='jik':
        grad1 = gRik
        grad2 = gRij

        term1 = (angle*(- (Rij*grad1) - (Rik*grad2)))/(Rij*Rik)
        term2 = (2*Ri - Rj - Rk)/(Rij * Rik)

        return term1 + term2 #checked
    
    elif indices=='kji':
        grad = gRij
        
        term1 = Rk - Rj
        term2 = angle*Rjk*grad

        return (term1 - term2)/(Rij*Rjk) #checked
    
    elif indices=='ikj':
        grad = gRik

        term1 = Rj - Rk
        term2 = angle*Rjk*grad

        return (term1 - term2)/(Rik*Rjk) #checked


@numba.jit(nopython=True)
def fcut(Rij, rcut): #checked
    return 0.5*(np.cos((np.pi*Rij)/rcut)+1)


@numba.jit(nopython=True)
def gradFcut(Rij, rcut): #checked
    return (-np.pi*np.sin((np.pi*Rij)/rcut))/(2*rcut)


@numba.jit(nopython=True)
def angular_gradients(size, atom_index, z, atom, charges, coods, cutoff_r, astep = 0.02, order=2,alength=160,a=2.0,grid1=None,grid2=None,angular_scaling1 = 4.0, angular_scaling2=2.4, zeta=1.0, lam=1.0, angular_scaling3=1.0):
    desc_size = 4*(order+1)
    arr=np.zeros((alength,desc_size))
    darr =np.zeros((alength,desc_size, 3))

    darr2 = np.zeros((alength,desc_size, size, 3))
    theta = 0.0
    z1 = z**0.8
    n1, n2 = angular_scaling1, angular_scaling2

    for i in range(alength):
        temp1, temp2 = grid1[i], grid2[i]
        num1, num2, num3, num4 = temp1
        costheta = num2
        f1 = np.zeros(desc_size)
        df1 = np.zeros((desc_size, 3))

        for j in range(size):
            if j!=atom_index:
                Ri, Rj = atom, coods[j]
                Rij = atom - coods[j]
                Rij_norm = np.linalg.norm(Rij)
                z2 = charges[j]**0.8

                df2 = np.zeros((desc_size, size, 3))

                if Rij_norm<cutoff_r:

                    for k in range(size):
                        if j!=k:
                            Rk = coods[k]
                            Rik = atom - coods[k]
                            Rik_norm = np.linalg.norm(Rik)

                            if Rik_norm!=0 and Rik_norm<cutoff_r:
                                z3 = charges[k]**0.8

                                Rkj = coods[k] - coods[j]

                                Rkj_norm = np.linalg.norm(Rkj)

                                costhetajik = np.minimum(1.0,np.maximum(np.dot(Rij,Rik)/(Rij_norm*Rik_norm),-1.0))
                                costhetakji = np.minimum(1.0,np.maximum(np.dot(Rij,Rkj)/(Rij_norm*Rkj_norm),-1.0))
                                costhetaikj = np.minimum(1.0,np.maximum(np.dot(-Rkj,Rik)/(Rkj_norm*Rik_norm),-1.0))

                                atm = Rij_norm*Rik_norm*Rkj_norm

                                charge = z1*z2*z3
                                x = costheta - costhetajik
                                x2 = x**2


                                gRij, gRik = gradRij(Ri, Rj, Rij_norm), gradRij(Ri, Rk, Rik_norm)
                                gRji, gRjk = gradRji(Rj, Ri, Rij_norm), gradRji(Rj, Rk, Rkj_norm)

                                fcutij, fcutik, fcutjk = fcut(Rij_norm, cutoff_r), fcut(Rik_norm, cutoff_r), fcut(Rkj_norm, cutoff_r)
                                fcut_tot = fcutij*fcutik*fcutjk
                                gfcutij, gfcutik, gfcutjk = gradFcut(Rij_norm, cutoff_r), gradFcut(Rik_norm, cutoff_r), gradFcut(Rkj_norm, cutoff_r)
                                gfcut_tot_i = (gfcutij*gRij*fcutik*fcutjk) + (gfcutik*gRik*fcutij*fcutjk) 
                                gfcut_tot_j = (gfcutij*gRji*fcutik*fcutjk) + (gfcutjk*gRjk*fcutik*fcutij)
                                #checked

                                gCosjik = gradCosAngi(Ri, Rj, Rk, Rij_norm, Rik_norm, Rkj_norm, gRij, gRik, costhetajik, 'jik')
                                gCoskji = gradCosAngi(Ri, Rj, Rk, Rij_norm, Rik_norm, Rkj_norm, gRij, gRik, costhetakji, 'kji')
                                gCosikj = gradCosAngi(Ri, Rj, Rk, Rij_norm, Rik_norm, Rkj_norm, gRij, gRik, costhetaikj, 'ikj')

                                gCosjik2 = gradCosAngj(Ri, Rj, Rk, Rij_norm, Rik_norm, Rkj_norm, gRji, gRjk, costhetajik, 'jik')
                                gCoskji2 = gradCosAngj(Ri, Rj, Rk, Rij_norm, Rik_norm, Rkj_norm, gRji, gRjk, costhetakji, 'kji')
                                gCosikj2 = gradCosAngj(Ri, Rj, Rk, Rij_norm, Rik_norm, Rkj_norm, gRji, gRjk, costhetaikj, 'ikj')
                                
                                exponent=np.exp(-a*x2)

                                g1_num = 1 + (3*costheta*costhetakji*costhetaikj)
                                denom1, denom2 = atm**n1, atm**n2
                                atm_gradi = exponent*(Rij_norm*gRik + Rik_norm*gRij)/(Rij_norm*Rik_norm)
                                atm_gradj = exponent*(Rij_norm*gRji + Rkj_norm*gRjk)/(Rij_norm*Rkj_norm)
                                
                                index=0
                                gradRhoi = 2*a*x*exponent*gCosjik
                                gradRhoj = 2*a*x*exponent*gCosjik2
                                gradi0 = gradRhoi - n1*atm_gradi
                                gradj0 = gradRhoj - n1*atm_gradj

                                #checked
                                termi1 = (costhetaikj*gCoskji + costhetakji*gCosikj)*exponent*costheta
                                termi2 = n2*atm_gradi*g1_num
                                gradi1 = termi1 + (gradRhoi*g1_num) - termi2
                                termj1 = (costhetaikj*gCoskji2 + costhetakji*gCosikj2)*exponent*costheta
                                termj2 = n2*atm_gradj*g1_num
                                gradj1 = termj1 + (gradRhoj*g1_num) - termj2
                                
                                for l in range(order+1):
                                    h = hermite_polynomial(x, l, a)
                                    pref = charge*exponent*h*fcut_tot
                                    g0 = num1/denom1
                                    ghermi = grad_hermite(x, l, gCosjik, a)
                                    ghermj = grad_hermite(x, l, gCosjik2, a)
                                    
                                    gradi = h*gradi0 + (ghermi*exponent)
                                    gradj = h*gradj0 + (ghermj*exponent)
                                    fn = pref*g0
                                    f1[index] += fn
                                    df1[index] += (gradi*g0*charge*fcut_tot) + (fn*gfcut_tot_i)
                                    df2[index][j] += (gradj*g0*charge*fcut_tot) + (fn*gfcut_tot_j)
                                    index+=1
                                    #checked
                                    
                                    gradi = h*gradi1 + (ghermi*exponent)
                                    gradj = h*gradj1 + (ghermi*exponent)
                                    fn = (g1_num*pref)/denom2
                                    f1[index] += fn
                                    df1[index] += ((charge*gradi*fcut_tot)/denom2) + (fn*gfcut_tot_i)
                                    df2[index][j] += ((charge*gradj*fcut_tot)/denom2) + (fn*gfcut_tot_j)
                                    index+=1

                                    g0 = num3/denom1
                                    ghermi = grad_hermite(x, l, gCosjik, a)
                                    ghermj = grad_hermite(x, l, gCosjik2, a)
                                    
                                    gradi = h*gradi0 + (ghermi*exponent)
                                    gradj = h*gradj0 + (ghermj*exponent)
                                    fn = pref*g0
                                    f1[index] += fn
                                    df1[index] += (gradi*g0*charge*fcut_tot) + (fn*gfcut_tot_i)
                                    df2[index][j] += (gradj*g0*charge*fcut_tot) + (fn*gfcut_tot_j)
                                    index+=1

                                    g0 = num4/denom1
                                    ghermi = grad_hermite(x, l, gCosjik, a)
                                    ghermj = grad_hermite(x, l, gCosjik2, a)
                                    
                                    gradi = h*gradi0 + (ghermi*exponent)
                                    gradj = h*gradj0 + (ghermj*exponent)
                                    fn = pref*g0
                                    f1[index] += fn
                                    df1[index] += (gradi*g0*charge*fcut_tot) + (fn*gfcut_tot_i)
                                    df2[index][j] += (gradj*g0*charge*fcut_tot) + (fn*gfcut_tot_j)
                                    index+=1

        darr2[i,:,:,:] = df2    
        
        theta+=0.02
        arr[i] = f1
        darr[i] = df1
    
    rep = [np.trapz(arr[:,i], dx = astep) for i in range(arr.shape[1])]

    darr2[:,:,atom_index,:] = darr    
    
    drep = np.zeros((desc_size, size, 3))

    for i in range(desc_size):
        for j in range(size):
            for k in range(3):
                drep[i][j][k] = np.trapz(darr2[:,i,j,k], dx=astep)
    return np.array(rep), drep    


@numba.jit(nopython=True)
def radial_gradients(size, z, atom, atom_index, charges, coods, cutoff_r, rlength,step_r, order = 2, a=1.0,eta=10.8,alpha=1.5,pow1=3.0,pow2=6.0):
    desc_size = 4*(order+1)
    arr = np.zeros((rlength,desc_size))
    darr = np.zeros((rlength, desc_size, 3))
    
    darr2 = np.zeros((rlength, desc_size, size, 3))
    r = 0
    z1 = z**0.8

    for i in range(rlength):
        f1 = np.zeros(desc_size)
        df1 = np.zeros((desc_size, 3))
        g1 = np.exp(-eta*r)
        g2 = 2.2508*((r+1)**pow2)
        g3 = np.exp(-alpha*r)
        g4 = 2.2508*((r+1)**pow1)
        
        for j in range(size):
            
            Ri, Rj = atom, coods[j]
            Rij = atom - coods[j]
            Rij_norm = np.linalg.norm(Rij)

            if Rij_norm!=0 and Rij_norm<cutoff_r:
                dist = Rij_norm
                x=r-dist
                charge = (charges[j]**0.8)*z1
                exponent=np.exp(-a*(x)**2)
                index = 0
                
                x2 = x**2
                #g3 = (1 - (alpha*x2))
                gradientRij = gradRij(Ri, Rj, Rij_norm)
                gradientRji = gradRji(Rj, Ri, Rij_norm)
                fcutrij = fcut(dist, cutoff_r)
                gradfcut = gradFcut(dist, cutoff_r)
                gradfcuti = gradfcut*gradientRij
                gradfcutj = gradfcut*gradientRji


                for k in range(order+1):
                    h = hermite_polynomial(x, k, a)
                    pref = charge*exponent*h

                    gradi0 = 2*pref*a*x*gradientRij
                    gradj0 = 2*pref*a*x*gradientRji

                    gradi1 = charge*exponent*grad_hermite(x, k, gradientRij, a)
                    gradj1 = charge*exponent*grad_hermite(x, k, gradientRji, a)

                    
                    fnr = pref*g1 
                    f1[index] += fnr*fcutrij
                    df1[index] += (gradi0 + gradi1)*g1*fcutrij + ((fnr*gradfcuti))
                    darr2[i,index,j,:] += (gradj0 + gradj1)*g1*fcutrij + (fnr*gradfcutj)
                    index+=1

                    
                    fnr = pref/g2
                    f1[index] += fnr*fcutrij
                    df1[index] += (gradi0 + gradi1)*fcutrij/(g2) + ((fnr*gradfcuti))
                    darr2[i,index,j,:] += (gradj0 + gradj1)*fcutrij/(g2) + (fnr*gradfcutj)
                    index+=1
                    
                    #fnr = pref*(g3**2)
                    #f1[index] += fnr*fcutrij
                    #df1[index] += ((gradi0 + gradi1)*(g3**2) + pref * (4*a*g3*x*gradientRij))*fcutrij + (fnr*gradfcuti)
                    #darr2[i,index,j,:] += ((gradj0 + gradj1)*(g3**2) + pref * (4*a*g3*x*gradientRji))*fcutrij + (fnr*gradfcutj)
                    #index+=1

                    fnr = pref*g3 
                    f1[index] += fnr*fcutrij
                    df1[index] += (gradi0 + gradi1)*g3*fcutrij + ((fnr*gradfcuti))
                    darr2[i,index,j,:] += (gradj0 + gradj1)*g3*fcutrij + (fnr*gradfcutj)
                    index+=1

                    fnr = pref/g4
                    f1[index] += fnr*fcutrij
                    df1[index] += (gradi0 + gradi1)*fcutrij/(g4) + ((fnr*gradfcuti))
                    darr2[i,index,j,:] += (gradj0 + gradj1)*fcutrij/(g4) + (fnr*gradfcutj)
                    index+=1
            
        r+=step_r
        arr[i] = f1
        darr[i] = df1

    rep = [np.trapz(arr[:,i], dx = step_r) for i in range(arr.shape[1])]

    darr2[:,:,atom_index,:] = darr 
    drep = np.zeros((desc_size, size, 3))

    for i in range(desc_size):
        for j in range(size):
            for k in range(3):
                drep[i][j][k] = np.trapz(darr2[:,i,j,k], dx=step_r)

    return np.array(rep), drep


@numba.jit(nopython=True)
def mbdf_with_gradients(charges,coods,grid1,grid2,rlength,alength,astep,order=3,nofcut=False,pad=29,step_r=0.1,cutoff_r=12,angular_scaling1=4.0,angular_scaling2=4.0,eta=10.8,alpha=1.5,pow1=3.0,pow2=6.0,a1 = 1.0, a2=2.0,zeta=1.0, lam=1.0, angular_scaling3=1.0,radial_only=False):
    """
    returns the local MBDF representation along with its gradients for a molecule
    """

    size = len(charges)
    nr, na = 4*(order+1), 4*(order+1)
    desc_size = nr+na
    
    rep = np.zeros((pad, desc_size))
    drep = np.zeros((pad, desc_size, pad, 3))
   
    for i in range(size):

        rep[i][:nr], drep[i,:nr,:size,:] = radial_gradients(size, charges[i], coods[i], i, charges, coods, cutoff_r, rlength, step_r, order, a1, eta, alpha, pow1, pow2)
        rep[i][nr:nr+na], drep[i,nr:nr+na,:size,:] = angular_gradients(size, i, charges[i], coods[i], charges, coods, cutoff_r, astep, order, alength, a2, grid1, grid2, angular_scaling1, angular_scaling2, zeta, lam, angular_scaling3)
    
    return rep, drep

                        
#@numba.jit(nopython=True)
def fourier_grid(step):
    
    angles = np.arange(0,np.pi,step)

    grid0 = np.cos(0.0*angles)
    grid1 = np.cos(1.0*angles)
    grid2 = np.cos(2.0*angles)
    grid3 = np.cos(3.0*angles)
    
    return [np.array([grid0, grid1, grid2, grid3]).T, grid1]


from joblib import Parallel, delayed


def generate_mbdf_train(nuclear_charges,coords,order=2,n_jobs=-1,pad=None,step_r=0.04,cutoff_r=8.0,step_a=0.02,a1 = 0.5, a2 = 0.5,angular_scaling=2.0,progress_bar=False,angular_scaling1=2.0,angular_scaling2=2.0,eta=5.0,alpha=1.5,pow1=5.0,pow2=2.8,zeta=1.0, lam=1.0, angular_scaling3=1.0,nofcut=False):
    
    assert nuclear_charges.shape[0] == coords.shape[0], "charges and coordinates array length mis-match"
    
    lengths, charges = [], []

    for i in range(len(nuclear_charges)):
        
        q, r = nuclear_charges[i], coords[i]
        
        assert q.shape[0] == r.shape[0], "charges and coordinates array length mis-match for molecule at index" + str(i)

        lengths.append(len(q))

        charges.append(q.astype(np.float64))

    if pad==None:
        pad = max(lengths)

    charges = np.array(charges)

    grid1,grid2 = fourier_grid(step_a)
    
    coords, cutoff_r = a2b*coords, a2b*cutoff_r

    rlength = int(cutoff_r/step_r) + 1
    alength = int(np.pi/step_a) + 1

    if progress_bar:
        from tqdm import tqdm
        mbdf = Parallel(n_jobs=n_jobs)(delayed(mbdf_with_gradients)(charge,cood,grid1,grid2,rlength,alength,step_a,order,nofcut,pad,step_r,cutoff_r,angular_scaling1,angular_scaling2,eta,alpha,pow1,pow2,a1,a2,zeta, lam, angular_scaling3) for charge, cood in tqdm(list(zip(charges, coords))))    
        
        A, dA = [], []
        for i in range(len(mbdf)):
            A.append(mbdf[i][0])
            dA.append(mbdf[i][1])
        A, dA = np.array(A), np.array(dA)
        
        norms = []

        for i in range(A.shape[2]):
            
            diff1 = np.abs(np.max(A[:,:,i])-np.min(A[:,:,i]))
            diff2 = np.abs(np.max(dA[:,:,i,:,:]) - np.min(dA[:,:,i,:,:]))

            if diff1!=0.0 and diff2!=0.0:
                A[:,:,i] = A[:,:,i]/diff1

                dA[:,:,i,:,:] = dA[:,:,i,:,:]/diff2

                norms.append([diff1, diff2])
            
            else:
                norms.append([1.0, 1.0])

        return A, dA, norms
    
    else:
        mbdf = Parallel(n_jobs=n_jobs)(delayed(mbdf_with_gradients)(charge,cood,grid1,grid2,rlength,alength,step_a,order,nofcut,pad,step_r,cutoff_r,angular_scaling1,angular_scaling2,eta,alpha,pow1,pow2,a1,a2,zeta, lam, angular_scaling3) for charge, cood in list(zip(charges, coords)))   
        
        A, dA = [], []
        for i in range(len(mbdf)):
            A.append(mbdf[i][0])
            dA.append(mbdf[i][1])
        A, dA = np.array(A), np.array(dA)
        
        norms = []

        for i in range(A.shape[2]):
            
            diff1 = np.abs(np.max(A[:,:,i])-np.min(A[:,:,i]))
            diff2 = np.abs(np.max(dA[:,:,i,:,:]) - np.min(dA[:,:,i,:,:]))

            if diff1!=0.0 and diff2!=0.0:
                A[:,:,i] = A[:,:,i]/diff1

                dA[:,:,i,:,:] = dA[:,:,i,:,:]/diff2

                norms.append([diff1, diff2])
            
            else:
                norms.append([1.0, 1.0])

        return A, dA, norms
    

def generate_mbdf_pred(nuclear_charges,coords, norms, order=2,n_jobs=-1,pad=None,step_r=0.04,cutoff_r=8.0,step_a=0.02,a1 = 0.5, a2 = 0.5,angular_scaling=2.0,progress_bar=False,angular_scaling1=2.0,angular_scaling2=2.0,eta=5.0,alpha=1.5,pow1=5.0,pow2=2.8,zeta=1.0, lam=1.0, angular_scaling3=1.0,nofcut=False):
    
    assert nuclear_charges.shape[0] == coords.shape[0], "charges and coordinates array length mis-match"
    
    lengths, charges = [], []

    for i in range(len(nuclear_charges)):
        
        q, r = nuclear_charges[i], coords[i]
        
        assert q.shape[0] == r.shape[0], "charges and coordinates array length mis-match for molecule at index" + str(i)

        lengths.append(len(q))

        charges.append(q.astype(np.float64))

    if pad==None:
        pad = max(lengths)

    charges = np.array(charges)

    grid1,grid2 = fourier_grid(step_a)
    
    coords, cutoff_r = a2b*coords, a2b*cutoff_r

    rlength = int(cutoff_r/step_r) + 1
    alength = int(np.pi/step_a) + 1

    if progress_bar:
        from tqdm import tqdm
        mbdf = Parallel(n_jobs=n_jobs)(delayed(mbdf_with_gradients)(charge,cood,grid1,grid2,rlength,alength,step_a,order,nofcut,pad,step_r,cutoff_r,angular_scaling1,angular_scaling2,eta,alpha,pow1,pow2,a1,a2,zeta, lam, angular_scaling3) for charge, cood in tqdm(list(zip(charges, coords))))    
        
        A, dA = [], []
        for i in range(len(mbdf)):
            A.append(mbdf[i][0])
            dA.append(mbdf[i][1])
        A, dA = np.array(A), np.array(dA)
        
        for i in range(len(norms)):
            
            diff1, diff2 = norms[i]

            if diff1!=0.0 and diff2!=0.0:
                A[:,:,i] = A[:,:,i]/diff1

                dA[:,:,i,:,:] = dA[:,:,i,:,:]/diff2

        return A, dA
    
    else:
        mbdf = Parallel(n_jobs=n_jobs)(delayed(mbdf_with_gradients)(charge,cood,grid1,grid2,rlength,alength,step_a,order,nofcut,pad,step_r,cutoff_r,angular_scaling1,angular_scaling2,eta,alpha,pow1,pow2,a1,a2,zeta, lam, angular_scaling3) for charge, cood in list(zip(charges, coords)))   
        
        A, dA = [], []
        for i in range(len(mbdf)):
            A.append(mbdf[i][0])
            dA.append(mbdf[i][1])
        A, dA = np.array(A), np.array(dA)
        
        for i in range(len(norms)):
            
            diff1, diff2 = norms[i]

            if diff1!=0.0 and diff2!=0.0:
                A[:,:,i] = A[:,:,i]/diff1

                dA[:,:,i,:,:] = dA[:,:,i,:,:]/diff2

        return A, dA
