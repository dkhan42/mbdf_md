import numpy as np
import numba
from copy import deepcopy

root2,ipi=2**0.5,np.pi*1j
half_rootpi=(np.pi**0.5)/2
c1,c2,c3=4.08858*(10**12),(np.pi**0.5)/2,(np.pi**0.5)*np.exp(-0.25)*1j/4
c4=-1j*(np.pi**0.5)*np.exp(-1/8)/(4*root2)
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
def generate_data(size,z,atom,charges,coods,cutoff_r=12.0):
    """
    returns 2 and 3-body internal coordinates
    """
    
    twob=np.zeros((size,2))
    threeb=np.zeros((size,size,5))
    z1=z**0.8

    for j in range(size):
        rij=atom-coods[j]
        rij_norm=np.linalg.norm(rij)

        if rij_norm!=0 and rij_norm<cutoff_r:
            z2=charges[j]**0.8
            twob[j]=rij_norm,z*charges[j]

            for k in range(size):
                if j!=k:
                    rik=atom-coods[k]
                    rik_norm=np.linalg.norm(rik)
            
                    if rik_norm!=0 and rik_norm<cutoff_r:
                        z3=charges[k]**0.8
                        
                        rkj=coods[k]-coods[j]
                        
                        rkj_norm=np.linalg.norm(rkj)
                        
                        threeb[j][k][0] = np.minimum(1.0,np.maximum(np.dot(rij,rik)/(rij_norm*rik_norm),-1.0))
                        threeb[j][k][1] = np.minimum(1.0,np.maximum(np.dot(rij,rkj)/(rij_norm*rkj_norm),-1.0))
                        threeb[j][k][2] = np.minimum(1.0,np.maximum(np.dot(-rkj,rik)/(rkj_norm*rik_norm),-1.0))
                        
                        atm = rij_norm*rik_norm*rkj_norm
                        
                        charge = z1*z2*z3
                        
                        threeb[j][k][3:] =  atm, charge

    return twob, threeb                        

@numba.jit(nopython=True)
def angular_integrals(size,threeb,style = 'hermite', order=2,alength=158,a=2,grid1=None,grid2=None,angular_scaling1 = 4.0, angular_scaling2=2.4, zeta=1.0, lam=1.0, angular_scaling3=1.0):
    """
    evaluates the 3-body functionals using the trapezoidal rule
    """
    #alength = alength*2
    arr=np.zeros((alength,desc_size))
    costheta=-1
    
    if style == 'hermite':
        for i in range(alength):
            num1,num2=grid1[i],grid2[i]
            f1 = np.zeros(desc_size)

            for j in range(size):

                for k in range(size):

                    if threeb[j][k][-1]!=0:

                        angle1,angle2,angle3,atm,charge=threeb[j][k]

                        x=costheta- angle1

                        exponent=np.exp(-a*x**2)
                        index=0

                        for l in range(order+1):
                            h = hermite_polynomial(x, l, a)
                            pref = charge*exponent*h
                            f1[index] += (pref*num1)/(atm**angular_scaling1)
                            index+=1
                            f1[index] += (pref*(1+(3*num2*angle1*angle2*angle3)))/(atm**angular_scaling2)
                            index+=1

                        #f3+=(charge*h2*exponent*(2**(1-zeta))*((1+(lam*angle1))**zeta))/(atm**angular_scaling3)
            costheta+=0.012658227848101266
            arr[i] = f1
            #arr[i]=f1,f2,f3
            #costheta+=0.012658227848101266
        trapz=[np.trapz(arr[:,i],dx = 0.012658227848101266) for i in range(arr.shape[1])]
    
    elif style == 'moments-centered':
        for i in range(alength):
            num1,num2=grid1[i],grid2[i]
            f1 = np.zeros(desc_size)

            for j in range(size):

                for k in range(size):

                    if threeb[j][k][-1]!=0:

                        angle1,angle2,angle3,atm,charge=threeb[j][k]

                        x=costheta- angle1

                        exponent=np.exp(-a*x**2)
                        index=0

                        for l in range(order+1):
                            #h = hermite_polynomial(x, l, a)
                            h = x**l
                            pref = charge*exponent*h
                            f1[index] += (pref*num1)/(atm**angular_scaling1)
                            index+=1
                            f1[index] += (pref*(1+(3*num2*angle1*angle2*angle3)))/(atm**angular_scaling2)
                            index+=1

                        #f3+=(charge*h2*exponent*(2**(1-zeta))*((1+(lam*angle1))**zeta))/(atm**angular_scaling3)
            costheta+=0.012658227848101266
            arr[i] = f1
        trapz=[np.trapz(arr[:,i],dx = 0.012658227848101266) for i in range(arr.shape[1])]

    return trapz


@numba.jit(nopython=True)
def radial_integrals(size,rlength,twob,step_r, order = 2,style = 'hermite', a=1,normalized=False,eta=10.8,alpha=1.5,pow1=3.0,pow2=6.0):
    """
    evaluates the 2-body functionals using the trapezoidal rule
    """
    arr=np.zeros((rlength,desc_size))
    r=0
    if style == 'hermite':
        for i in range(rlength):
            f1 = np.zeros(desc_size)
            for j in range(size):
                if twob[j][-1]!=0:
                    dist,charge=twob[j]
                    x=r-dist
                    if normalized==True:
                        norm=(erfunc(dist)+1)*half_rootpi
                        exponent=np.exp(-a*(x)**2)/norm
                    else:
                        exponent=np.exp(-a*(x)**2)
                    index=0
                    for k in range(order+1):
                        h = hermite_polynomial(x, k, a)
                        pref = charge*exponent*h
                        f1[index] += pref*np.exp(-eta*r) - pref/(2.2508*(r+1)**pow1) 
                        index+=1
                        f1[index] += pref/(2.2508*(r+1)**pow2)
                        index+=1
                        f1[index] += pref*np.exp(-alpha*r)
                        index+=1
            r+=step_r
            arr[i]=f1
        trapz=[np.trapz(arr[:,i],dx=step_r) for i in range(arr.shape[1])]
    elif style == 'moments-centered':
        for i in range(rlength):
            f1 = np.zeros(desc_size)
            for j in range(size):
                if twob[j][-1]!=0:
                    dist,charge=twob[j]
                    x=r-dist
                    if normalized==True:
                        norm=(erfunc(dist)+1)*half_rootpi
                        exponent=np.exp(-a*(x)**2)/norm
                    else:
                        exponent=np.exp(-a*(x)**2)
                    index=0
                    for k in range(order+1):
                        h = x**k
                        pref = charge*exponent*h
                        f1[index] += pref*np.exp(-eta*r) - pref/(2.2508*(r+1)**pow1) 
                        index+=1
                        f1[index] += pref/(2.2508*(r+1)**pow2)
                        index+=1
                        f1[index] += pref*np.exp(-alpha*r)
                        index+=1
            r+=step_r
            arr[i]=f1
        trapz=[np.trapz(arr[:,i],dx=step_r) for i in range(arr.shape[1])]
    elif style == 'moments-mean':
        for i in range(rlength):
            center = np.zeros(3)
            for j in range(size):
                if twob[j][-1]!=0:
                    dist,charge=twob[j]
                    x=r-dist
                    if normalized==True:
                        norm=(erfunc(dist)+1)*half_rootpi
                        exponent=np.exp(-a*(x)**2)/norm
                    else:
                        exponent=np.exp(-a*(x)**2)
                    pref = charge*exponent
                    center[0] += pref*np.exp(-eta*r) - pref/(2.2508*(r+1)**pow1)
                    center[1] += pref/(2.2508*(r+1)**pow2)
                    center[2] += pref*np.exp(-alpha*r)
            r+=step_r
            arr[i][:3] = center
        r0 = np.asarray([np.trapz(arr[:,i], dx=step_r) for i in range(3)])
        for i in range(rlength):
            f1 = np.zeros(3*(order))
            for j in range(size):
                if twob[j][-1]!=0:
                    dist,charge=twob[j]
                    x=r-dist
                    if normalized==True:
                        norm=(erfunc(dist)+1)*half_rootpi
                        exponent=np.exp(-a*(x)**2)/norm
                    else:
                        exponent=np.exp(-a*(x)**2)
                    index=3
                    for k in range(1,order+1):
                        h = (r - r0[0])**k
                        pref = charge*exponent*h
                        f1[index] += pref*np.exp(-eta*r) - pref/(2.2508*(r+1)**pow1)
                        index+=1
                        h = (r - r0[1])**k
                        pref = charge*exponent*h
                        f1[index] = pref/(2.2508*(r+1)**pow2)
                        index+=1
                        h = (r - r0[2])**k
                        pref = charge*exponent*h
                        f1[index] = pref*np.exp(-alpha*r)
                        index+=1
            r+= step_r
            arr[i][3:] = f1
        trapz=[np.trapz(arr[:,i],dx=step_r) for i in range(3,arr.shape[1])]
        trapz.extend(r0)

    return trapz


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
def gradCosAngi(Ri, Rj, Rk, Rij, Rik, Rjk, gRij, gRik, angle, indices='jik'):
    
    if indices=='jik':
        grad1 = gRik
        grad2 = gRij

        term1 = (angle*(- (Rij*grad1) - (Rik*grad2)))/(Rij*Rik)
        term2 = (2*Ri - Rj - Rk)/(Rij * Rik)

        return term1 + term2
    
    elif indices=='kji':
        grad = gRij
        
        term1 = Rk - Rj
        term2 = angle*Rjk*grad

        return (term1 - term2)/(Rij*Rjk)
    
    elif indices=='ikj':
        grad = gRik

        term1 = Rj - Rk
        term2 = angle*Rjk*grad

        return (term1 - term2)/(Rik*Rjk)


@numba.jit(nopython=True)
def angular_gradients(size, atom_index, z, atom, charges, coods, cutoff_r, astep = 0.01, order=2,alength=200,a=2,grid1=None,grid2=None,angular_scaling1 = 4.0, angular_scaling2=2.4, zeta=1.0, lam=1.0, angular_scaling3=1.0):
    desc_size = 3*(order+1)
    arr=np.zeros((alength,desc_size))
    darr =np.zeros((alength,desc_size, 3))

    darr2 = np.zeros((alength,desc_size, size, 3))
    costheta=-1
    z1 = z**0.8
    n1, n2, n3 = angular_scaling1, angular_scaling2, angular_scaling3
    a2 = a**2

    g2_pref = 2**(1-lam)
    g2_pref_grad = g2_pref*zeta*lam

    #print(darr.shape, darr2.shape)

    for i in range(alength):
        num1,num2=grid1[i],grid2[i]
        f1 = np.zeros(desc_size)
        df1 = np.zeros((desc_size, 3))

        for j in range(size):
            Ri, Rj = atom, coods[j]
            Rij = atom - coods[j]
            Rij_norm = np.linalg.norm(Rij)
            z2 = charges[j]**0.8
            
            df2 = np.zeros((desc_size, size, 3))
            
            if Rij_norm!=0 and Rij_norm<cutoff_r:

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

                            #charge = z1*z2*z3
                            charge=1

                            x = costheta - costhetajik
                            x2 = x**2

                            exponent=np.exp(-a*x2)
                            index=0

                            gRij, gRik = gradRij(Ri, Rj, Rij_norm), gradRij(Ri, Rk, Rik_norm)
                            gRji, gRjk = gradRji(Rj, Ri, Rij_norm), gradRji(Rj, Rk, Rkj_norm)

                            fcutij, fcutik, fcutjk = fcut(Rij_norm, cutoff_r), fcut(Rik_norm, cutoff_r), fcut(Rkj_norm, cutoff_r)
                            fcut_tot = fcutij*fcutik*fcutjk
                            gfcutij, gfcutik, gfcutjk = gradFcut(Rij_norm, cutoff_r), gradFcut(Rik_norm, cutoff_r), gradFcut(Rkj_norm, cutoff_r)
                            gfcut_tot_i = (gfcutij*gRij*fcutik*fcutjk) + (gfcutik*gRik*fcutij*fcutjk) 
                            gfcut_tot_j = (gfcutij*gRji*fcutik*fcutjk) + (gfcutjk*gRjk*fcutik*fcutij)
                            
                            gCosjik = gradCosAngi(Ri, Rj, Rk, Rij_norm, Rik_norm, Rkj_norm, gRij, gRik, costhetajik, 'jik')
                            gCoskji = gradCosAngi(Ri, Rj, Rk, Rij_norm, Rik_norm, Rkj_norm, gRij, gRik, costhetakji, 'kji')
                            gCosikj = gradCosAngi(Ri, Rj, Rk, Rij_norm, Rik_norm, Rkj_norm, gRij, gRik, costhetaikj, 'ikj')
                            
                            gradRhoi = -2*a*x*exponent*gCosjik

                            gradi0 = gradRhoi - (n1*exponent*(Rij_norm*gRik + Rik_norm*gRij)/(Rij_norm*Rik_norm))
                            termi1 = (costhetaikj*gCoskji + costhetakji*gCosikj)*exponent*costheta
                            g1_num = 1 + (3*costheta*costhetakji*costhetaikj)
                            termi2 = (n2*exponent*g1_num)*(Rij_norm*gRik + Rik_norm*gRij)/(Rij_norm*Rik_norm)
                            gradi1 = termi1 + (gradRhoi*g1_num) - termi2

                            denom1, denom2, denom3 = atm**n1, atm**n2, np.exp(n3*(Rij_norm**2 + Rik_norm**2 + Rkj_norm**2)) 
                            
                            gCosjik2 = gradCosAngj(Ri, Rj, Rk, Rij_norm, Rik_norm, Rkj_norm, gRji, gRjk, costhetajik, 'jik')
                            gCoskji2 = gradCosAngj(Ri, Rj, Rk, Rij_norm, Rik_norm, Rkj_norm, gRji, gRjk, costhetakji, 'kji')
                            gCosikj2 = gradCosAngj(Ri, Rj, Rk, Rij_norm, Rik_norm, Rkj_norm, gRji, gRjk, costhetaikj, 'ikj')
                            
                            gradRhoj = -2*a*x*exponent*gCosjik2
                            
                            gradj0 = gradRhoj - (n1*exponent*(Rij_norm*gRji + Rkj_norm*gRjk)/(Rij_norm*Rkj_norm))
                            termj1 = (costhetaikj*gCoskji2 + costhetakji*gCosikj2)*exponent*costheta
                            termj2 = (n2*exponent*g1_num)*(Rij_norm*gRjk + Rkj_norm*gRji)/(Rij_norm*Rkj_norm)
                            gradj1 = termj1 + (gradRhoj*g1_num) - termj2
                            
                            g2_brac = (1+(lam*costhetajik))**(zeta)
                            g2_num = g2_pref*g2_brac
                            g2_num_grad = g2_pref_grad*(g2_brac/(1+(lam*costhetajik)))
                            termi1 = -(2*n3*Rij_norm*gRij) - (2*n3*Rik_norm*gRik)
                            gradi2 = ((gradRhoi*g2_num)+ (termi1*exponent) + (g2_num_grad*gCosjik*exponent))/denom3
                            termj1 = -(2*n3*Rij_norm*gRji) - (2*n3*Rkj_norm*gRjk)
                            gradj2 = ((gradRhoj*g2_num) + (termj1*exponent) + (g2_num_grad*gCosjik2*exponent))/denom3

                            for l in range(order+1):
                                h = hermite_polynomial(x, l, a)
                                pref = charge*exponent*h*fcut_tot
                                g0 = num1/denom1
                                
                                if l==0:
                                    gradi = gradi0
                                    gradj = gradj0
                                elif l==1:
                                    gradi = h*gradi0 - (2*a*gCosjik)
                                    gradj = h*gradj0 - (2*a*gCosjik2)
                                elif l==2:
                                    gradi = h*gradi0 + (8*a2*(-x)*gCosjik)
                                    gradj = h*gradj0 + (8*a2*(-x)*gCosjik2)
                                elif l==3:
                                    temp = 2*a2*x2 - 1
                                    gradi = h*gradi0 + (12*a*temp*gCosjik)
                                    gradj = h*gradj0 + (12*a*temp*gCosjik2)

                                fn = pref*g0
                                f1[index] += fn
                                df1[index] += (gradi*g0*charge*fcut_tot) + (fn*gfcut_tot_i)
                                df2[index][j] += (gradj*g0*charge*fcut_tot) + (fn*gfcut_tot_j)
                                index+=1

                                if l==0:
                                    gradi = gradi1
                                    gradj = gradj1
                                elif l==1:
                                    gradi = h*gradi1 - (2*a*gCosjik)
                                    gradj = h*gradj1 - (2*a*gCosjik2)
                                elif l==2:
                                    gradi = h*gradi1 + (8*a2*(-x)*gCosjik)
                                    gradj = h*gradj1 + (8*a2*(-x)*gCosjik2)
                                elif l==3:
                                    temp = 2*a2*x2 - 1
                                    gradi = h*gradi1 + (12*a*temp*gCosjik)
                                    gradj = h*gradj1 + (12*a*temp*gCosjik2)
                                    
                                fn = (g1_num*pref)/denom2
                                f1[index] += fn
                                df1[index] += ((charge*gradi*fcut_tot)/denom2) + (fn*gfcut_tot_i)
                                df2[index][j] += ((charge*gradj*fcut_tot)/denom2) + (fn*gfcut_tot_j)
                                index+=1

                                if l==0:
                                    gradi = gradi2
                                    gradj = gradj2
                                elif l==1:
                                    gradi = h*gradi2 - (2*a*gCosjik)
                                    gradj = h*gradj2 - (2*a*gCosjik2)
                                elif l==2:
                                    gradi = h*gradi2 + (8*a2*(-x)*gCosjik)
                                    gradj = h*gradj2 + (8*a2*(-x)*gCosjik2)
                                elif l==3:
                                    temp = 2*a2*x2 - 1
                                    gradi = h*gradi2 + (12*a*temp*gCosjik)
                                    gradj = h*gradj2 + (12*a*temp*gCosjik2)
                                
                                fn = pref*g2_num/denom3
                                f1[index] += fn
                                df1[index] += (gradi*charge*fcut_tot) + (fn*gfcut_tot_i)
                                df2[index][j] += (gradj*charge*fcut_tot) + (fn*gfcut_tot_j)
                                index+=1

        darr2[i,:,:,:] = df2    
        
        costheta+=astep
        arr[i] = f1
        darr[i] = df1
    
    rep = [np.trapz(arr[:,i], dx = astep) for i in range(arr.shape[1])]

    darr2[:,:,atom_index,:] = darr    

    #drep = [[np.trapz(darr[:,i,0], dx=0.01), np.trapz(darr[:,i,1], dx=0.01), np.trapz(darr[:,i,2], dx=0.01)]
    #         for i in range(arr.shape[1])]
    
    drep = np.zeros((desc_size, size, 3))

    for i in range(desc_size):
        for j in range(size):
            for k in range(3):
                drep[i][j][k] = np.trapz(darr2[:,i,j,k], dx=astep)
    #print(np.array(rep).shape, drep.shape)
    return np.array(rep), drep    


@numba.jit(nopython=True)
def fcut(Rij, rcut):
    return 0.5*(np.cos((np.pi*Rij)/rcut)+1)

@numba.jit(nopython=True)
def gradFcut(Rij, rcut):
    return (-np.pi*np.sin((np.pi*Rij)/rcut))/(2*rcut)

@numba.jit(nopython=True)
def radial_gradients(size, z, atom, atom_index, charges, coods, cutoff_r, rlength,step_r, order = 2, a=1,eta=10.8,alpha=1.5,pow1=3.0,pow2=6.0):
    desc_size = 3*(order+1)
    arr = np.zeros((rlength,desc_size))
    darr = np.zeros((rlength, desc_size, 3))
    
    darr2 = np.zeros((rlength, desc_size, size, 3))
    r = 0
    z1 = z**0.8
    a2 = a**2

    for i in range(rlength):
        f1 = np.zeros(desc_size)
        df1 = np.zeros((desc_size, 3))
        

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
                gradientRij = gradRij(Ri, Rj, Rij_norm)
                gradientRji = gradRji(Rj, Ri, Rij_norm)
                fcutrij = fcut(dist, cutoff_r)
                gradfcut = gradFcut(dist, cutoff_r)
                gradfcuti = gradfcut*gradientRij
                gradfcutj = gradfcut*gradientRji

                for k in range(order+1):
                    h = hermite_polynomial(x, k, a)
                    pref = charge*exponent*h*fcutrij
                    if k==0:
                        pref2 = a*charge
                        grad = 2*pref2*gradientRij*exponent*dist
                        gradj = 2*pref2*gradientRji*exponent*dist
                    elif k==1:
                        pref2 = -2*a*charge
                        grad = pref2*gradientRij*exponent*(2*a*(x2) - 1)
                        gradj = pref2*gradientRji*exponent*(2*a*(x2) - 1)
                    elif k==2:
                        pref2 = -4*a2*charge
                        grad = pref2*gradientRij*exponent*(-x*(2*a*x2 - 3))
                        gradj = pref2*gradientRji*exponent*(-x*(2*a*x2 - 3))
                    elif k==3:
                        pref2 = 4*a2*charge
                        grad = pref2*gradientRij*exponent*(2*a*x2*(3 - 2*a*x2) + 6*a*x2 - 3)
                        gradj = pref2*gradientRji*exponent*(2*a*x2*(3 - 2*a*x2) + 6*a*x2 - 3)
                    g = np.exp(-eta*r) - 1/(2.2508*((r+1)**pow1))
                    fnr = pref*g 
                    f1[index] += fnr*fcutrij
                    df1[index] += (grad*g*fcutrij) + (fnr*gradfcuti)
                    darr2[i,index,j,:] += (gradj*g*fcutrij) + (fnr*gradfcutj)
                    index+=1
                    g = 2.2508*((r+1)**pow2)
                    fnr = pref/g
                    f1[index] += fnr*fcutrij
                    df1[index] += (grad*fcutrij/g) + (fnr*gradfcuti)
                    darr2[i,index,j,:] += (gradj*fcutrij/g) + (fnr*gradfcutj)
                    index+=1
                    g = np.exp(-alpha*r)
                    fnr = pref*g
                    f1[index] += fnr*fcutrij
                    df1[index] += (grad*g*fcutrij) + (fnr*gradfcuti)
                    darr2[i,index,j,:] += (gradj*g*fcutrij) + (fnr*gradfcutj)
                    index+=1
        
        r+=step_r
        arr[i] = f1
        darr[i] = df1

    rep = [np.trapz(arr[:,i], dx = step_r) for i in range(arr.shape[1])]
    
#    drep = [[np.trapz(darr[:,i,0], dx=step_r), np.trapz(darr[:,i,1], dx=step_r), np.trapz(darr[:,i,2], dx=step_r)]
#             for i in range(arr.shape[1])]
    #print(darr.shape, darr2.shape)
    darr2[:,:,atom_index,:] = darr 
    drep = np.zeros((desc_size, size, 3))

    for i in range(desc_size):
        for j in range(size):
            for k in range(3):
                drep[i][j][k] = np.trapz(darr2[:,i,j,k], dx=step_r)
    #print(np.array(rep).shape, drep.shape)
    return np.array(rep), drep


@numba.jit(nopython=True)
def mbdf_with_gradients(charges,coods,grid1,grid2,rlength,alength,astep,order=3,pad=29,step_r=0.1,cutoff_r=12,angular_scaling1=4.0,angular_scaling2=4.0,eta=10.8,alpha=1.5,pow1=3.0,pow2=6.0,a1=1.0,a2=1.2,zeta=1.0, lam=1.0, angular_scaling3=1.0,radial_only=False):
    """
    returns the local MBDF representation along with its gradients for a molecule
    """
    size = len(charges)
    nr, na = 3*(order+1), 3*(order+1)
    if radial_only==False:
        desc_size = nr+na
        rep = np.zeros((pad, desc_size))
        drep = np.zeros((pad, desc_size, pad, 3))
    
        for i in range(size):
            #try:
            #print(rep[i][:nr].shape, drep[i,:nr,:size,:].shape)
            rep[i][:nr], drep[i,:nr,:size,:] = radial_gradients(size, charges[i], coods[i], i, charges, coods, cutoff_r, rlength, step_r, order, a1, eta, alpha, pow1, pow2)
            #print(rep[i][nr:nr+na].shape, drep[i,nr:nr+na,:size,:].shape)
            rep[i][nr:nr+na], drep[i,nr:nr+na,:size,:] = angular_gradients(size, i, charges[i], coods[i], charges, coods, cutoff_r, astep, order, alength, a2, grid1, grid2, angular_scaling1, angular_scaling2, zeta, lam, angular_scaling3)
            #except:
            #    temp3, temp4 = radial_gradients(size, charges[i], coods[i], i, charges, coods, cutoff_r, rlength, step_r, order, a1, eta, alpha, pow1, pow2)
            #
            #    temp1, temp2 = angular_gradients(size, i, charges[i], coods[i], charges, coods, cutoff_r, astep, order, alength, a2, grid1, grid2, angular_scaling1, angular_scaling2, zeta, lam, angular_scaling3)
            #    print(temp3.shape, temp4.shape, temp1.shape, temp2.shape)
            #    break
        #drep[i,nr:nr+na,:size,:] = temp
    else:
        desc_size = nr
        rep = np.zeros((pad, desc_size))
        drep = np.zeros((pad, desc_size, pad, 3))
    
        for i in range(size):
            #try:
            #print(rep[i][:nr].shape, drep[i,:nr,:size,:].shape)
            rep[i][:nr], drep[i,:nr,:size,:] = radial_gradients(size, charges[i], coods[i], i, charges, coods, cutoff_r, rlength, step_r, order, a1, eta, alpha, pow1, pow2)

    return rep, drep


@numba.jit(nopython=True)
def mbdf_local(charges,coods,grid1,grid2,rlength,alength,style,order=3,pad=29,step_r=0.1,cutoff_r=12,angular_scaling1=4.0,angular_scaling2=4.0,eta=10.8,alpha=1.5,pow1=3.0,pow2=6.0,a1=1.0,a2=1.2,zeta=1.0, lam=1.0, angular_scaling3=1.0):
    """
    returns the local MBDF representation for a molecule
    """
    size = len(charges)
    mat=np.zeros((pad,20))
    nr = desc_size
    na = desc_size
    
    assert size > 1, "No implementation for monoatomics"

    if size>2:
        for i in range(size):

            twob,threeb = generate_data(size,charges[i],coods[i],charges,coods,cutoff_r)

            mat[i][:nr] = radial_integrals(size,rlength,twob,step_r,order=order,style=style,eta=eta,alpha=alpha,pow1=pow1,pow2=pow2,a=a1)     

            mat[i][nr:nr+na] = angular_integrals(size,threeb,style,order,alength,grid1=grid1,grid2=grid2,a=a2,angular_scaling1=angular_scaling1,angular_scaling2=angular_scaling2,zeta=zeta, lam=lam, angular_scaling3=angular_scaling3)

    elif size==2:
        z1, z2, rij = charges[0]**0.8, charges[1]**0.8, coods[0]-coods[1]
        
        pref, dist = z1*z2, np.linalg.norm(rij)
        
        twob = np.array([[pref, dist], [pref, dist]])
        
        mat[0][:4] = radial_integrals(size,rlength,twob,step_r)

        mat[1][:4] = mat[0][:4]

    return mat


def mbdf_global(charges,coods,asize,rep_size,keys,grid1,grid2,rlength,alength,step_r=0.1,cutoff_r=12,angular_scaling=2.4):
    """
    returns the flattened, bagged MBDF feature vector for a molecule
    """
    elements = {k:[[],k] for k in keys}

    size = len(charges)

    for i in range(size):
        elements[charges[i]][0].append(coods[i])

    mat, ind = np.zeros((rep_size,6)), 0

    assert size > 1, "No implementation for monoatomics"

    if size>2:

        for key in keys:
            
            num = len(elements[key][0])
            
            if num!=0:
                bags = np.zeros((num,6))
                
                for j in range(num):
                    twob,threeb = generate_data(size,key,elements[key][0][j],charges,coods,cutoff_r)

                    bags[j][:4] = radial_integrals(size,rlength,twob,step_r)     

                    bags[j][4:] = angular_integrals(size,threeb,alength,grid1=grid1,grid2=grid2,angular_scaling=angular_scaling)

                mat[ind:ind+num] = -np.sort(-bags,axis=0)
                
            ind += asize[key]
    
    elif size == 2:

        for key in keys:
            
            num = len(elements[key][0])
            
            if num!=0:
                bags = np.zeros((num,6))
                
                for j in range(num):
                    z1, z2, rij = charges[0]**0.8, charges[1]**0.8, coods[0]-coods[1]
        
                    pref, dist = z1*z2, np.linalg.norm(rij)

                    twob = np.array([[pref, dist], [pref, dist]])
                    
                    bags[j][:4] = radial_integrals(size,rlength,twob,step_r)     

                mat[ind:ind+num] = -np.sort(-bags,axis=0)
                
            ind += asize[key]

    return mat
                        

@numba.jit(nopython=True)
def fourier_grid(a,b,c,d,step):
    
    angles = np.arange(0,np.pi,step)
    
    grid1 = np.cos(angles)
    grid2 = np.cos(2*angles)
    grid3 = np.cos(3*angles)
    
    #return (3+(100*grid1)+(-200*grid2)+(-164*grid3),grid1)
    return (a+(b*grid1)+(c*grid2)+(d*grid3),grid1)


@numba.jit(nopython=True)
def normalize_with_gradients(A, dA,normal='mean'):
    """
    normalizes the functionals based on the given method
    """
    
    A_temp = np.zeros(A.shape)
    
    if normal=='mean':
        for i in range(A.shape[2]):
            
            avg = np.mean(A[:,:,i])
            

            if avg!=0.0:
                A_temp[:,:,i] = A[:,:,i]/avg
                dA[:,:,i,:,:] = dA[:,:,i,:,:]/avg
            
            else:
                pass
   
    elif normal=='min-max':
        for i in range(A.shape[2]):
            
            diff1 = np.abs(np.max(A[:,:,i])-np.min(A[:,:,i]))
            #diff2 = np.abs(np.max(dA[:,:,i,:,:]) - np.min(dA[:,:,i,:,:]))

            #if diff1!=0.0 and diff2!=0.0:
            #    A_temp[:,:,i] = A[:,:,i]/diff1
#
            #    dA[:,:,i,:,:] = dA[:,:,i,:,:]/diff2

            if diff1!=0.0:
                A_temp[:,:,i] = A[:,:,i]/diff1

                dA[:,:,i,:,:] = dA[:,:,i,:,:]/diff1
            
            else:
                pass
    
    return A_temp, dA


@numba.jit(nopython=True)
def normalize(A,normal='mean'):
    """
    normalizes the functionals based on the given method
    """
    
    A_temp = np.zeros(A.shape)
    
    if normal=='mean':
        for i in range(A.shape[2]):
            
            avg = np.mean(A[:,:,i])

            if avg!=0.0:
                A_temp[:,:,i] = A[:,:,i]/avg
            
            else:
                pass
   
    elif normal=='min-max':
        for i in range(A.shape[2]):
            
            diff = np.abs(np.max(A[:,:,i])-np.min(A[:,:,i]))
            
            if diff!=0.0:
                A_temp[:,:,i] = A[:,:,i]/diff
            
            else:
                pass
    
    return A_temp


from joblib import Parallel, delayed

def generate_mbdf(nuclear_charges,coords,radial_only=False,order=3,local=True, gradients=True,n_jobs=-1,pad=None,step_r=0.04,cutoff_r=8.0,step_a=0.01,a1=1.0,a2=2.0,angular_scaling=4,normalized='min-max',progress_bar=False,angular_scaling1=4,angular_scaling2=4,eta=10.8,alpha=1.5,pow1=3,pow2=6,a=1,b=1,c=1,d=1,zeta=1.0, lam=1.0, angular_scaling3=1.0):
    """
    Generates the local MBDF representation arrays for a set of given molecules

    :param nuclear_charges: array of arrays of nuclear_charges for all molecules in the dataset
    :type nuclear_charges: numpy array NxM, where N is the number of molecules and M is the number of atoms (can be different for each molecule)
    :param coords : array of arrays of input coordinates of the atoms
    :type coords: numpy array NxMx3, where N is the number of molecules and M is the number of atoms (can be different for each molecule)
    ordering of the molecules in the nuclear_charges and coords arrays should be consistent
    :param n_jobs: number of cores to parallelise the representation generation over. Default value is -1 which uses all available cores in the system
    :type n_jobs: integer
    :param pad: Number of atoms in the largest molecule in the dataset. Can be left to None and the function will calculate it using the nuclear_charges array
    :type pad: integer
    :param step_r: radial step length in Angstrom
    :type step_r: float
    :param cutoff_r: local radial cutoff distance for each atom
    :type cutoff_r: float
    :param step_a: angular step length in Radians
    :type step_a: float
    :param angular_scaling: scaling of the inverse distance weighting used in the angular functionals
    :type : float
    :param normalized: type of normalization to be applied to the functionals. Available options are 'min-max' and 'mean'. Can be turned off by passing False
    :type : string
    :param progress: displays a progress bar for representation generation process. Requires the tqdm library
    :type progress: Bool

    :return: NxPadx6 array containing Padx6 dimensional MBDF matrices for the N molecules
    """
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

    #rlength = int(cutoff_r/step_r) + 1
    #alength = int(2/step_a) + 1

    grid1,grid2 = fourier_grid(a,b,c,d,step_a)
    
    coords, cutoff_r = a2b*coords, a2b*cutoff_r

    rlength = int(cutoff_r/step_r) + 1
    alength = int(2/step_a) + 1

    if local:
        if progress_bar:

            from tqdm import tqdm

            if gradients:
                mbdf = Parallel(n_jobs=n_jobs)(delayed(mbdf_with_gradients)(charge,cood,grid1,grid2,rlength,alength,step_a,order,pad,step_r,cutoff_r,angular_scaling1,angular_scaling2,eta,alpha,pow1,pow2,a1,a2,zeta, lam, angular_scaling3, radial_only) for charge, cood in tqdm(list(zip(charges, coords))))    
                
                #mbdf = []
                #for charge, cood in tqdm(list(zip(charges, coords))):
                #    mbdf.append(mbdf_with_gradients(charge,cood,grid1,grid2,rlength,alength,style,step_a,order,pad,step_r,cutoff_r,angular_scaling1,angular_scaling2,eta,alpha,pow1,pow2,a1,a2,zeta, lam, angular_scaling3))

                rep, drep = [], []
                for i in range(len(mbdf)):
                    rep.append(mbdf[i][0])
                    drep.append(mbdf[i][1])
                rep, drep = np.array(rep), np.array(drep)
                
                if normalized==False:
                    return rep, drep
                else:
                    return normalize_with_gradients(rep, drep, normal=normalized)
            
            else:
                
                mbdf = Parallel(n_jobs=n_jobs)(delayed(mbdf_local)(charge,cood,grid1,grid2,rlength,alength,order,pad,step_r,cutoff_r,angular_scaling1,angular_scaling2,eta,alpha,pow1,pow2,a1,a2,zeta, lam, angular_scaling3) for charge,cood in tqdm(list(zip(charges,coords))))
                mbdf = np.array(mbdf)
                
                if normalized==False:
                    return mbdf
                
                else:
                    return normalize(mbdf, normal=normalized)


        else:
            if gradients:
                mbdf = Parallel(n_jobs=n_jobs)(delayed(mbdf_with_gradients)(charge,cood,grid1,grid2,rlength,alength,step_a,order,pad,step_r,cutoff_r,angular_scaling1,angular_scaling2,eta,alpha,pow1,pow2,a1,a2,zeta, lam, angular_scaling3, radial_only) for charge, cood in list(zip(charges, coords)))    
                
                rep, drep = [], []
                for i in range(len(mbdf)):
                    rep.append(mbdf[i][0])
                    drep.append(mbdf[i][1])
                rep, drep = np.array(rep), np.array(drep)
                
                if normalized==False:
                    return rep, drep
                else:
                    return normalize_with_gradients(rep, drep, normal=normalized)
            
            else:
                
                mbdf = Parallel(n_jobs=n_jobs)(delayed(mbdf_local)(charge,cood,grid1,grid2,rlength,alength,order,pad,step_r,cutoff_r,angular_scaling1,angular_scaling2,eta,alpha,pow1,pow2,a1,a2,zeta, lam, angular_scaling3) for charge,cood in list(zip(charges,coords)))
                mbdf = np.array(mbdf)
                
                if normalized==False:
                    return mbdf
                
                else:
                    return normalize(mbdf, normal=normalized)
        
    else:
        keys = np.unique(np.concatenate(charges))

        asize = {key:max([(mol == key).sum() for mol in charges]) for key in keys}

        rep_size = sum(asize.values())

        if progress_bar==True:

            from tqdm import tqdm    
            arr = Parallel(n_jobs=n_jobs)(delayed(mbdf_global)(charge,cood,asize,rep_size,keys,grid1,grid2,rlength,alength,step_r,cutoff_r,angular_scaling) for charge,cood in tqdm(list(zip(charges,coords))))

        else:
            arr = Parallel(n_jobs=n_jobs)(delayed(mbdf_global)(charge,cood,asize,rep_size,keys,grid1,grid2,rlength,alength,step_r,cutoff_r,angular_scaling) for charge,cood in zip(charges,coords))

        arr = np.array(arr)

        if normalized==False:

            mbdf = np.array([mat.ravel(order='F') for mat in arr])
            
            return mbdf

        else:

            arr = normalize(arr,normal=normalized)

            mbdf = np.array([mat.ravel(order='F') for mat in arr])
            
            return mbdf


@numba.jit(nopython=True)
def wKDE(rep,bin,bandwidth,kernel,scaling=False):
    """
    returns the weighted kernel density estimate for a given array and bins
    """
    if kernel=='gaussian':
        if scaling=='root':
            a = bin.reshape(-1,1)-rep
            
            basis = np.exp(-(a**2)/bandwidth)
            
            k = (np.sqrt(np.abs(rep)))*basis
            
            return np.sum(k,axis=1)

        else:
            a = bin.reshape(-1,1)-rep
            
            basis = np.exp(-(a**2)/bandwidth)
            
            return np.sum(basis,axis=1)

    elif kernel=='laplacian':
        if scaling=='root':
            a = bin.reshape(-1,1)-rep
            
            basis = np.exp(-(np.abs(a))/bandwidth)
            
            k = (np.abs(rep))*basis
            
            return np.sum(k,axis=1)

        else:
            a = bin.reshape(-1,1)-rep
            
            basis = np.exp(-(np.abs(a))/bandwidth)
            
            return np.sum(basis,axis=1)


def density_estimate(reps,nuclear_charges,keys,bin,bandwidth,kernel='gaussian',scaling='root'):
    """
    returns the density functions of MBDF functionals for a set of given molecules.
    """
    
    size=len(bin)
    big_rep=np.zeros((reps.shape[0],size*len(keys)))
    

    if kernel=='gaussian':
        for i in range(len(nuclear_charges)):

            for j,k in enumerate(keys):
                ii = np.where(nuclear_charges[i] == k)[0]

                if len(ii)!=0:
                    big_rep[i,j*size:(j+1)*size]=wKDE(reps[i][ii]/k,bin,bandwidth,kernel,scaling)

                else:
                    big_rep[i,j*size:(j+1)*size]=np.zeros(size)

    return big_rep


def generate_df(mbdf,nuclear_charges,bw=0.07,binsize=0.2,kernel='gaussian'):
    """
    Generates the Density of Functionals representation for a given set of molecules. Requires their MBDF arrays as input
    
    :param mbdf: array of arrays containing the MBDF representation matrices for all molecules in the dataset
    :type mbdf: numpy array, output of the generate_mbdf function can be directly used here
    :param nuclear_charges: array of arrays of nuclear_charges for all molecules in the dataset, should be in the same order as in the MBDF arrays
    :type nuclear_charges: numpy array NxM, where N is the number of molecules and M is the number of atoms (can be different for each molecule)
    :param bw: the bandwidth hyper-parameter of the kernel density estimate
    :type bw: float
    :param binsize: grid-spacing used for discretizing the density function
    :type binsize: float
    :param kernel: kernel function to be used in the kernel density estimation
    :type kernel: string

    :return: NxM array containing the M dimensional representation vectors for N molecules
    """
    fs=mbdf.shape[-1]

    reps=[10*mbdf[:,:,i]/(np.max(np.abs(mbdf[:,:,i]))) for i in range(fs)]
    
    keys=np.unique(np.concatenate(nuclear_charges))
    
    bin=np.arange(-10,10,binsize)
    
    gridsize=len(keys)*len(bin)
    
    kde=np.zeros((mbdf.shape[0],gridsize*fs))
    
    for i in range(fs):
        kde[:,i*gridsize:(i+1)*gridsize]=density_estimate(reps[i],nuclear_charges,keys,bin,bw,kernel)
    
    return kde

@numba.jit(nopython=True)
def generate_CM(cood,charges,pad):
    size=len(charges)
    cm=np.zeros((pad,pad))
    for i in range(size):
        for j in range(size):
            if i==j:
                cm[i,j]=0.5*(charges[i]**(2.4))
            else:
                dist=np.linalg.norm(cood[i,:]-cood[j,:])
                
                cm[i,j]=(charges[i]*charges[j])/dist
    summation = np.array([sum(x**2) for x in cm])
    sorted_mat = cm[np.argsort(summation)[::-1,],:]    
    return sorted_mat.ravel()


from math import comb,cos
from itertools import combinations, product


def generate_bob(elements,coords,n_jobs=-1,asize={'C': 7, 'H': 16, 'N': 3, 'O': 3, 'S': 1}):
    """
    generates the Bag of Bonds representation
    :param elements: array of arrays of chemical element symbols for all molecules in the dataset
    :type elements: numpy array NxM, where N is the number of molecules and M is the number of atoms (can be different for each molecule)
    :param coords: array of arrays of input coordinates of the atoms
    :type coords: numpy array NxMx3, where N is the number of molecules and M is the number of atoms (can be different for each molecule)
    :param n_jobs: number of cores to parallelise the representation generation over. Default value is -1 which uses all cores in the system
    :type n_jobs: integer
    :param asize: The maximum number of atoms of each element type supported by the representation
    :type asize: dictionary

    :return: NxD array of D-dimensional BoB vectors for the N molecules
    :rtype: numpy array
    """
    from tqdm import tqdm

    bob_arr = Parallel(n_jobs=n_jobs)(delayed(bob)(atoms,coods,asize) for atoms,coods in tqdm(list(zip(elements,coords))))

    return np.array(bob_arr)

def bob(atoms,coods, asize={'C': 7, 'H': 16, 'N': 3, 'O': 3, 'S': 1}):
    keys=list(asize.keys()) 
    elements={'C':[[],6],'H':[[],1],'N':[[],7],'O':[[],8],'F':[[],9],'P':[[],15],'S':[[],16],'Cl':[[],17],'Br':[[],35],'I':[[],53]}
    for i in range(len(atoms)):
        elements[atoms[i]][0].append(coods[i])
    bob=[]
    for key in keys:
        num=len(elements[key][0])
        if num!=0:
            bag=np.zeros((asize[key]))
            bag[:num]=0.5*(elements[key][1]**2.4)
            bag=-np.sort(-bag)
            bob.extend(bag)
            for j in range(i,len(keys)):
                if i==j:
                    z=elements[key][1]
                    bag=np.zeros((comb(asize[key],2)))
                    vec=[]
                    for (r1,r2) in combinations(elements[key][0],2):
                        vec.append(z**2/np.linalg.norm(r1-r2))
                    bag[:len(vec)]=vec
                    bag=-np.sort(-bag)
                    bob.extend(bag)
                elif (i!=j) and (len(elements[keys[j]][0])!=0):
                    z1,z2=elements[key][1],elements[keys[j]][1]
                    bag=np.zeros((asize[key]*asize[keys[j]]))
                    vec=[]
                    for (r1,r2) in product(elements[key][0],elements[keys[j]][0]):
                        vec.append(z1*z2/np.linalg.norm(r1-r2))
                    bag[:len(vec)]=vec
                    bag=-np.sort(-bag)
                    bob.extend(bag)
                else:
                    bob.extend(np.zeros((asize[key]*asize[keys[j]])))
        else:
            bob.extend(np.zeros((asize[key])))
            for j in range(i,len(keys)):
                if i==j:
                    bob.extend(np.zeros((comb(asize[key],2))))
                else:
                    bob.extend(np.zeros((asize[key]*asize[keys[j]])))
    return np.array(bob) 

from scipy.spatial.distance import cityblock, euclidean
from scipy.stats import wasserstein_distance

def get_delta_local_kernel(A,B,Q1,Q2,sigma,kernel='laplacian'):
    
    n1, n2 = A.shape[0], B.shape[0]

    assert n1 == Q1.shape[0], "charges and representation array length mis-match"
    assert n2 == Q2.shape[0], "charges and representation array length mis-match"

    K = 0
    
    if kernel == 'laplacian':

        for i in range(n1):
            k=0
            for j in range(n2):
                q1, q2 = Q1[i], Q2[j]

                if q1==q2:
                    dist = cityblock(A[i],B[j])
                    k += np.exp(-dist/sigma)
            K += k

    elif kernel == 'gaussian':

        for i in range(n1):
            k=0
            for j in range(n2):
                q1, q2 = Q1[i], Q2[j]

                if q1==q2:
                    dist = euclidean(A[i],B[j])
                    k += np.exp(-dist/sigma)
            K += k

    elif kernel == 'wasserstein':

        for i in range(n1):
            k=0
            for j in range(n2):
                q1, q2 = Q1[i], Q2[j]

                if q1==q2:
                    dist = wasserstein_distance(A[i],B[j])
                    k += np.exp(-dist/sigma)
            K += k
    
    return K

def get_min_local_kernel(A,B,Q1,Q2,sigma,kernel='laplacian'):
    
    n1, n2 = A.shape[0], B.shape[0]

    assert n1 == Q1.shape[0], "charges and representation array length mis-match"
    assert n2 == Q2.shape[0], "charges and representation array length mis-match"

    K1, K2 = 0, 0
    
    if kernel == 'laplacian':

        for i in range(n1):
            k= []
            for j in range(n2):
                q1, q2 = Q1[i], Q2[j]

                if q1==q2:
                    dist = cityblock(A[i],B[j])
                    k.append(np.exp(-dist/sigma))
            K1 += max(k)
        for i in range(n2):
            k= []
            for j in range(n1):
                q1, q2 = Q1[j], Q2[i]

                if q1==q2:
                    dist = cityblock(A[j],B[i])
                    k.append(np.exp(-dist/sigma))
            K2 += max(k)
        return min([K1,K2])

    elif kernel == 'gaussian':

        for i in range(n1):
            k= []
            for j in range(n2):
                q1, q2 = Q1[i], Q2[j]

                if q1==q2:
                    dist = euclidean(A[i],B[j])
                    k.append(np.exp(-dist/sigma))
            K1 += max(k)
        for i in range(n2):
            k= []
            for j in range(n1):
                q1, q2 = Q1[j], Q2[i]

                if q1==q2:
                    dist = euclidean(A[j],B[i])
                    k.append(np.exp(-dist/sigma))
            K2 += max(k)
        return min([K1,K2])

    elif kernel == 'wasserstein':

        for i in range(n1):
            k= []
            for j in range(n2):
                q1, q2 = Q1[i], Q2[j]

                if q1==q2:
                    dist = wasserstein_distance(A[i],B[j])
                    k.append(np.exp(-dist/sigma))
            K1 += max(k)
        for i in range(n2):
            k= []
            for j in range(n1):
                q1, q2 = Q1[j], Q2[i]

                if q1==q2:
                    dist = wasserstein_distance(A[j],B[i])
                    k.append(np.exp(-dist/sigma))
            K2 += max(k)

        return min([K1,K2])

def get_max_local_kernel(A,B,Q1,Q2,sigma,kernel='laplacian'):
    
    n1, n2 = A.shape[0], B.shape[0]

    assert n1 == Q1.shape[0], "charges and representation array length mis-match"
    assert n2 == Q2.shape[0], "charges and representation array length mis-match"

    K1, K2 = 0, 0
    
    if kernel == 'laplacian':

        for i in range(n1):
            k= []
            for j in range(n2):
                q1, q2 = Q1[i], Q2[j]

                if q1==q2:
                    dist = cityblock(A[i],B[j])
                    k.append(np.exp(-dist/sigma))
            K1 += max(k)
        for i in range(n2):
            k= []
            for j in range(n1):
                q1, q2 = Q1[j], Q2[i]

                if q1==q2:
                    dist = cityblock(A[j],B[i])
                    k.append(np.exp(-dist/sigma))
            K2 += max(k)
        return max([K1,K2])

    elif kernel == 'gaussian':

        for i in range(n1):
            k= []
            for j in range(n2):
                q1, q2 = Q1[i], Q2[j]

                if q1==q2:
                    dist = euclidean(A[i],B[j])
                    k.append(np.exp(-dist/sigma))
            K1 += max(k)
        for i in range(n2):
            k= []
            for j in range(n1):
                q1, q2 = Q1[j], Q2[i]

                if q1==q2:
                    dist = euclidean(A[j],B[i])
                    k.append(np.exp(-dist/sigma))
            K2 += max(k)
        return max([K1,K2])

    elif kernel == 'wasserstein':

        for i in range(n1):
            k= []
            for j in range(n2):
                q1, q2 = Q1[i], Q2[j]

                if q1==q2:
                    dist = wasserstein_distance(A[i],B[j])
                    k.append(np.exp(-dist/sigma))
            K1 += max(k)
        for i in range(n2):
            k= []
            for j in range(n1):
                q1, q2 = Q1[j], Q2[i]

                if q1==q2:
                    dist = wasserstein_distance(A[j],B[i])
                    k.append(np.exp(-dist/sigma))
            K2 += max(k)

        return max([K1,K2])
