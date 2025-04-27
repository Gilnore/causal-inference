# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 01:55:46 2024

@author: robin

1. Nagel, D., Diez, G. & Stock, G. Accurate estimation of the normalized mutual information of multidimensional data. The Journal of Chemical Physics 161, 054108 (2024).
2. Czyż, P., Grabowski, F., Vogt, J. E., Beerenwinkel, N. & Marx, A. Beyond Normal: On the Evaluation of Mutual Information Estimators. Preprint at https://doi.org/10.48550/arXiv.2306.11078 (2023).
3. Baboukani, P. S., Graversen, C., Alickovic, E. & Østergaard, J. Estimating Conditional Transfer Entropy in Time Series using Mutual Information and Non-linear Prediction. Entropy 22, 1124 (2020).
4. Kraskov, A., Stoegbauer, H. & Grassberger, P. Estimating Mutual Information. Phys. Rev. E 69, 066138 (2004).
5. Li, R. et al. Mining Spatio-Temporal Relations via Self-Paced Graph Contrastive Learning. Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining 936–944 (2022) doi:10.1145/3534678.3539422.
6. Witter, J. & Houghton, C. Nearest-Neighbours Estimators for Conditional Mutual Information. Preprint at https://doi.org/10.48550/arXiv.2403.00556 (2024).

test to see if psi(mean(n)) is better than mean(psi(n)) for giving global entropies
"""
import numpy as np
import ball_tree as bt
from sklearn.cross_decomposition import CCA
from scipy.special import psi, binom
from numba import float64, int64, boolean
from typing import Tuple
import numba as nb
import numba_scipy




@nb.vectorize([float64(float64)])
def vabs(x:float)->float:
    """
    #to be able to use vectorized version of abs in numba

    Parameters
    ----------
    x : float
    Returns
    -------
    float
    """
    
    X = abs(x)
    return X

@nb.njit([(float64[:,:],float64[:])], parallel=True)
def cmax(x:np.ndarray,X:np.ndarray):
    """
    #equivalent to np.amax on axis 1

    Parameters
    ----------
    x : np.ndarray
        array to find max of.
    X : np.ndarray
        the return array to modify.

    Returns
    -------
    None.

    """
    
    for i in nb.prange(x.shape[0]):
        X[i] = np.max(x[i])

@nb.njit([(float64[:,:],float64[:,:])], parallel=True)
def choose_by_min(x:np.ndarray,X:np.ndarray):
    """
    #this function picks the columns of X in each row, 
    #based on min of x in each row
    #take the absolute minimum on the column

    Parameters
    ----------
    x : np.ndarray
        measure.
    X : np.ndarray
        values.

    Returns
    -------
    X_fin : TYPE
        picked.
    """
    
    args = np.zeros(x.shape[0],dtype = np.int64)
    for i in nb.prange(x.shape[0]):
        j = np.argmin(np.abs(x[i]))
        args[i] = j
    #pick elements based on the min
    X_fin = np.zeros(x.shape[0])
    for k in nb.prange(x.shape[0]):
        X_fin[k] = X[k,args[k]]
    return X_fin

@nb.njit
def cadd(x:np.ndarray):
    """
    np.add with axis 1 for numba

    Parameters
    ----------
    x : np.ndarray
    Returns
    -------
    X : np.ndarray
    """
    X = np.zeros(x.shape[0])
    for i in range(x.shape[1]):
        X += x[:,i]
    return X

@nb.njit
def nbhstack(X:list)->np.ndarray:
    """
    see np.hstack, but for 2d and higher arrays

    Parameters
    ----------
    X : list
    Returns
    -------
    np.ndarray

    """
    if len(X) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    shape1 = 0
    shape0 = X[0].shape[0]
    for xi in range(len(X)):
        shape1+= X[xi].shape[1]
        assert shape0 == X[xi].shape[0], "can't stack inhomogenous arrays"
    final = np.zeros((shape0,shape1))
    last_col = 0
    for i in range(len(X)):
        shape = X[i].shape[1]
        final[:,last_col:last_col+shape] = X[i]
        last_col+=shape
    return np.ascontiguousarray(final)

@nb.njit
def nbvstack(X:list)->np.ndarray:
    """
    see np.vstack, but for 2d and higher arrays

    Parameters
    ----------
    X : list
    Returns
    -------
    np.ndarray

    """
    if len(X) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    shape1 = X[0].shape[0]
    shape0 = len(X)
    final = np.zeros((shape0,shape1))
    for i in range(len(X)):
        assert shape1 == X[i].shape[0], "can't stack inhomogenous arrays"
        final[i] = X[i]
    return np.ascontiguousarray(final)

@nb.vectorize([float64(float64)])
def vpsi(x:float)->float:
    """
    vectorized digamma function for numba

    Parameters
    ----------
    x : float
    Returns
    -------
    float

    """
    x = max(1,x)
    p = psi(np.float64(x))
    if p == np.inf:
        p = 0
    elif p == np.nan:
        p = 0
    return p

@nb.vectorize([float64(float64,float64)])
def vbinom(x:float,y:float)->float:
    b = binom(x,y)
    return b

@nb.njit
def H(X:np.ndarray,k:int, local:bool=False)->float:
    """
    the shannon entropy

    Parameters
    ----------
    X : np.ndarray
    k : int
        knn parameter.
    local : bool, optional
        return local entropies. The default is False.

    Returns
    -------
    float
        the entropy.

    """
    N = float(X.shape[0])
    dx = X.shape[1]
    space = bt.iterative_construct(X,metric="chebyshev")
    idx,dist = bt.query(X,X,space,k+1)
    dk = dist[:,k]
    k = float(k)
    avg_psi = psi(N) - psi(k) + (dx-1)/k
    eps =  np.log2((dk / (np.mean(dk**dx)**(1/dx))))
    if local:
         return avg_psi + dx*eps
    else:
        return np.array([avg_psi + dx*np.mean(eps)])
    
    
@nb.njit(parallel=True)
def I_nb(X:list, 
         k:int, 
         local=False, 
         normalize:bool=False)->Tuple[np.ndarray,np.ndarray]:
    """
    the mutual information by ksg method

    Parameters
    ----------
    X : list
        the collection of arrays to find mi of.
    k : int
        knn parameter.
    local : TYPE, optional
        return local mi. The default is False.
    normalize : bool, optional
        return normalized mi. The default is False.

    Returns
    -------
    TYPE
        MI or local MI.
    TYPE
        estimation error estimate.
    """
    #this might be used in other functions that need jit decoration
    #the entropy I(X1:...:Xn)
    Xjoint = nbhstack(X)
    xjoint = bt.iterative_construct(Xjoint,metric="chebyshev") #joint tree
    joint_idxs, joint_dist= bt.query(Xjoint,Xjoint,xjoint, k=k+1) #find kth
    joint_kth = Xjoint[joint_idxs[:,k]]
    N = Xjoint.shape[0]
    dist = joint_dist[:,k]
    joint_psi =  psi(float(k)) + (len(X) - 1)* psi(float(N)) - (Xjoint.shape[1]-1)/k
    joint_psi1 = psi(float(k)) + (len(X) - 1)* psi(float(N))
    avg_psi = np.zeros((N,len(X)))
    avg_psi1 = np.zeros((N,len(X)))
    shapes = np.array([xj.shape[1] for xj in X])
    ksg1err = np.zeros((N,len(X)))
    ksg2err = np.zeros((N,len(X)))
    for i in nb.prange(shapes.shape[0]):
        idx = np.sum(shapes[:i])
        Xi = X[i]
        xi = bt.iterative_construct(Xi,metric="chebyshev")
        dim = Xi.shape[1]
        marg_kth = joint_kth[:, idx:idx+dim]
        xi_dists = vabs(Xi - marg_kth)
        xidist = np.zeros(Xi.shape[0])
        cmax(xi_dists,xidist)
        ksg1err[:,i] = (dist**2-xidist*dist)/(dist**2)
        #area at each point in joint space is (dist**2-xidist*dist)

        count, _ = bt.query_radius(Xi, Xi, xi, xidist, count_only=True)
        count = nbvstack(count)
        count = count.reshape(count.shape[0],-1)
        count1, _ = bt.query_radius(Xi, Xi, xi, dist, count_only=True)
        count1 = nbvstack(count1)
        ksg2err[:,i] = ((dim/count**.5 - dim/count1**.5)*N**.5).flatten()
        #as k->inf, count -> N. stat error is dim/root(N)
        #so for each k up to N, the percentage error of count/k
        #is dim / root(count/k) = dim / root(count) / root(N)
        #difference in stat erro then occur in the count
        avg_psi[:,i] = (vpsi(count) - (dim-1)/(count))[:,0]
        avg_psi1[:,i] = vpsi(count1)[:,0]
    
    avg_psi = cadd(avg_psi)
    avg_psi1 = cadd(avg_psi1)
    ksg1err = cadd(ksg1err)
    ksg2err = cadd(ksg2err)
    I0 = joint_psi - avg_psi
    I1 = joint_psi1 - avg_psi1
    err =  nbvstack([ksg1err,ksg2err]).T 
    I = nbvstack([I1,I0]).T
    I = choose_by_min(err,I)

    err = choose_by_min(err,err)
    if normalize:
        entropies = np.zeros(shapes.shape[0])
        for i in nb.prange(shapes.shape[0]):
            entropies[i] = H(X[i],k)[0]
        normalizer = np.prod(entropies)**(1/shapes.shape[0])
        I /= normalizer
    if local:
        return I, err
    else:
        I = np.mean(I)
        return np.array([I]), np.array([np.mean(err)])

@nb.njit
def KSG_TE_nb(Yt:np.ndarray,
              X:np.ndarray,
              Y:np.ndarray,
              k:int=10,
              local:bool=False,
              normalize:bool=False
              )->Tuple[np.ndarray,np.ndarray]:
    """
    find the conditional mutual information or transfer entropy by KSG

    Parameters
    ----------
    Yt : np.ndarray
        target.
    X : np.ndarray
        source.
    Y : np.ndarray
        target history or excluded info.
    k : int, optional
        knn parameter. The default is 10.
    local : bool, optional
        return local entropies. The default is False.
    normalize : bool, optional
        return normalized entropies. The default is False.

    Returns
    -------
    np.ndarray
        the entropy either at each point in time or an average.
        returned as a 2d array when local else 1d array
    TYPE
        the error on prediction.

    """
    assert k < len(Yt), "not enough data"
    #concatonate Z to Y if one needs conditional TE
    XY = nbhstack([X,Y])
    YtY = nbhstack([Yt,Y])
    XYtY = nbhstack([X,YtY])
    N = XY.shape[0]
    mxy = XY.shape[1]
    myty = YtY.shape[1]
    mxyty = XYtY.shape[1]
    my = Y.shape[1]
    #we always use the same amount of lags for all spaces
    lags = X.shape[-1]
    #the spaces are made outside of this function
    #construct KDtree in each space and find kth neighbout
    xyty = bt.iterative_construct(XYtY,metric="chebyshev")
    yty = bt.iterative_construct(YtY,metric="chebyshev")
    y = bt.iterative_construct(Y,metric="chebyshev")
    xy = bt.iterative_construct(XY,metric="chebyshev")
    #distances is calculated in the three variable joint space
    
    xyty_idx, xyty_dist = bt.query(XYtY,XYtY,xyty,k=k+1)
    kth_idx = xyty_idx[:,k]
    kth_dist = xyty_dist[:,k]
    
    xyty_k = XYtY[kth_idx] #the neighbours
    #we need to compute the marginal distances of the joint neighbour
    xk = xyty_k[:,:lags]#marginalized x points picked from joint space
    ytyk = xyty_k[:,lags:]
    yk = xyty_k[:,lags+Yt.shape[1]:]
    xyk = nbhstack((xk,yk))
    #computing distances of joint space kth neighbour in marginal spaces for KSG2
    xydistk = np.zeros(XY.shape[0])
    xydist = vabs(XY-xyk)
    cmax(xydist,xydistk)
    err1 = (kth_dist**2 - (xydistk*kth_dist))/kth_dist**2

    ytydistk = np.zeros(YtY.shape[0])
    ytydist = vabs(YtY - ytyk)
    cmax(ytydist,ytydistk)
    err1 += (kth_dist**2 - (ytydistk*kth_dist))/kth_dist**2
    
    ydistk = np.zeros(Y.shape[0])
    ydist = vabs(Y-yk)
    cmax(ydist,ydistk)
    err1 += (kth_dist**2 - (ydistk*kth_dist))/kth_dist**2

    nxy, _ = bt.query_radius(XY, XY, xy, xydistk, count_only=True)
    nxy = nbvstack(nxy)
    ny, _ = bt.query_radius(Y, Y, y, ydistk, count_only=True)
    ny = nbvstack(ny)
    nyty, _ = bt.query_radius(YtY,YtY,yty, ytydistk, count_only=True)
    nyty = nbvstack(nyty)
    #the numbers are obtained in their respective marginal spaces
    nxy1, _ = bt.query_radius(XY, XY, xy, kth_dist, count_only=True)
    nxy1 = nbvstack(nxy1)
    err2 = (mxy/nxy**.5 - mxy/nxy1**.5)*N**.5
    
    ny1, _ = bt.query_radius(Y, Y, y, kth_dist, count_only=True)
    ny1 = nbvstack(ny1)
    err2 += (my/ny**.5 - my/ny1**.5)*N**.5
    
    nyty1, _ = bt.query_radius(YtY,YtY,yty, kth_dist, count_only=True)
    nyty1 = nbvstack(nyty1)
    err2 += (myty/nyty**.5 - myty/nyty1**.5)*N**.5
    
    err2 = err2.flatten()
    
    k=float(k)
    TE = psi(k) - (mxyty - 1)/k + (vpsi(ny) - (my-1)/ny) - (vpsi(nyty) - (myty-1)/nyty) - (vpsi(nxy) - (mxy-1)/nxy)
    TE1 = psi(k) + vpsi(ny1) - vpsi(nyty1) - vpsi(nxy1)

    err = nbvstack([err1,err2]).T
    TE_comp = nbhstack((TE1,TE))
    TE_fin = choose_by_min(err,TE_comp)
    err = choose_by_min(err,err)
    if normalize:
        mi = I_nb([Yt, X], int(k))[0]
        TE_fin/=mi
    if local:
        return TE_fin, err
    else:
        return np.array([np.mean(TE_fin)]), np.array([np.mean(err)])

# @dd.timeit

@nb.njit
def NN_CMI(Yt:np.ndarray,
          X:np.ndarray,
          Y:np.ndarray, 
          tolerance:int=2,
          local:bool=False, 
          normalize:bool=False)->Tuple[float,int]:
    """
    the KSG method of finding mutual information or transfer entropy
    with the knn parameter optimized to minimize KSG error
    
    Parameters
    ----------
    Yt : np.ndarray
        target array.
    X : np.ndarray
        source array.
    Y : np.ndarray
        target history or information to exclude.
    tolerance : int, optional
        error tolorance on k. The default is 2.
    local : bool, optional
        return local entropies. The default is False.
    normalize : bool, optional
        normalized entropies. The default is False.

    Returns
    -------
    Tuple[float,int]
        entropy and optimal k for ksg.

    """
    N,my = X.shape
    guess_k = int(np.ceil(N**.5))
    a = guess_k/2
    b = guess_k*1.5
    invphi = (5**.5 - 1)/2
    tolerance = max(1/invphi,tolerance)
    while b - a > tolerance:
        c = int(np.floor(b - (b - a) * invphi)) #new lower bound
        d = int(np.ceil(a + (b - a) * invphi)) #new upper bound

        if Y.shape[1]<1:
            Yt = np.ascontiguousarray(Yt)
            X = np.ascontiguousarray(X)
            _,fc = I_nb([Yt,X],k=c,local=False,normalize=normalize)
            _,fd = I_nb([Yt,X],k=d,local=False,normalize=normalize)
        else:
            Yt = np.ascontiguousarray(Yt)
            X = np.ascontiguousarray(X)
            Y = np.ascontiguousarray(Y)
            _,fc = KSG_TE_nb(Yt,X,Y,k=c,local=False,normalize=normalize)
            _,fd = KSG_TE_nb(Yt,X,Y,k=d,local=False,normalize=normalize)

        if fc[0] < fd[0]: #the function to the lower bound side is smaller
            b = d #elimiate the side to the upper bound
        else:  # f(c) > f(d) to find the maximum
            a = c #elimiate the other side
    k_opm = int(np.ceil((b + a) / 2)) #take the middle
    if Y.shape[1]<1:
        I,err = I_nb([Yt,X],k=k_opm,local=local,normalize=normalize)
    else:
        I, err = KSG_TE_nb(Yt,X,Y,k=k_opm,local=local,normalize=normalize)
    return I, k_opm


@nb.njit(
    [float64[:](float64[:,:],
            float64[:,:],
            int64,
            boolean)],
          parallel=True)
def MRS(target:np.ndarray,
        sources:np.ndarray,
        k:int,
        local:bool=False)->float:
    """
    find the mean root squared prediction error of knn using sources to predict
    target

    Parameters
    ----------
    target : np.ndarray
        to be predicted.
    sources : np.ndarray
        knn pool.
    k : int
        knn parameter.
    local : bool, optional
        each point in time (number of rows) or mean error. The default is False.

    Returns
    -------
    float or array of floats
        the error of prediction.

    """
    target_hat = np.zeros(target.shape)
    sause = bt.iterative_construct(sources)
    #find points in sources close
    idxs,dist = bt.query(sources,sources,sause,k=k+1)
    idxs = idxs[:,1:] #remove self column
    #find points in target occuring at the same time
    for i in range(k):
        target_hat += target[idxs[:,i]]
    target_hat /= k
    res = np.zeros(target.shape[0])
    for i in nb.prange(target.shape[0]):
        diff = (target[i] - target_hat[i])
        res[i] = np.dot(diff,diff)
    if local:
        return res
    else:
        return np.array([np.mean(res)])

@nb.njit(parallel=True)
def apply_mimrs_to_rows(target:np.ndarray,
                        sources:np.ndarray,
                        shapes:np.ndarray,
                        k:int,
                        l:float)->np.ndarray:
    t = np.ascontiguousarray(target)
    res = np.zeros(len(shapes))
    for j in nb.prange(len(shapes)):
        shape = shapes[j]
        have_been = np.sum(shapes[:j])
        source = sources[:, have_been:have_been+shape]
        s = np.ascontiguousarray(source)
        X = [t,s]
        I,err = I_nb(X,k, normalize=False)
        mrs = MRS(t,s, k, False)[0]
        res[j] = (1-l)*I[0] - l*mrs
    ind = np.argmax(res)
    chosen_shape = shapes[ind]
    chosen_ind = np.sum(shapes[:ind])
    return (chosen_ind,chosen_ind+chosen_shape, ind)

@nb.njit(parallel=True)
def apply_temrs_to_rows(target:np.ndarray,
                        sources:np.ndarray,
                        shapes:np.ndarray,
                        emb:np.ndarray,
                        k:int,
                        l:float,
                        use_dk:bool=False)->Tuple[int,int,int]:
    
    e = np.ascontiguousarray(emb)
    t = np.ascontiguousarray(target)
    res = np.zeros(len(shapes))
    for j in nb.prange(len(shapes)):
        shape = shapes[j]
        have_been = np.sum(shapes[:j])
        #dies on loop 2 here
        source = sources[:, have_been:have_been+shape]
        s = np.ascontiguousarray(source)
        if use_dk:
            I = DK_TE(t,s,e,local=False)
        else:
            I,err = KSG_TE_nb(t,s,e,k,local=False,normalize=False)
        U = nbhstack([e,s])
        mrs = MRS(t, U, k, False)[0]
        res[j] = (1-l)*I[0] - l*mrs
    ind = np.argmax(res)
    chosen_shape = shapes[ind]
    chosen_ind = np.sum(shapes[:ind])
    # chosen = sources[:, chosen_ind:chosen_ind+chosen_shape]
    return (chosen_ind,chosen_ind+chosen_shape, ind)

@nb.njit
def rank_by_MRS(target:np.ndarray,
                sources:np.ndarray,
                shapes:np.ndarray,
                emb:np.ndarray,
                k:int = 10,
                l:float = .75,
                use_dk:bool=False)->Tuple[int,int,int]:
    """
    rank the sources by information to target excuding information already included
    in the embeddings, and their regressive prediction errors
    
    Parameters
    ----------
    target : np.ndarray
    sources : np.ndarray
        candidates vectors stacked horrizontally by rows
    shapes : np.ndarray
        shapes for the multivariate vector sources, partitioning its widths.
    emb : np.ndarray
        what is already known
    k : int, optional
        Knn parameter. The default is 10.
    l : float, optional
        blend of information to error ratio. The default is .75.

    Returns
    -------
    Tuple[int,int,int]
        the start to finish idx of sources, and the idx of shape of that source.

    """
    
    if emb.shape[1] < 1:
        top_pick = apply_mimrs_to_rows(target, 
                                       sources, 
                                       shapes, 
                                       k, 
                                       l)
    else:
        top_pick = apply_temrs_to_rows(target, 
                                       sources, 
                                       shapes, 
                                       emb, 
                                       k, 
                                       l,
                                       use_dk)
    return top_pick

@nb.njit
def construct_candidates(arrays:tuple,
                         max_lag:int,
                         noise_level:float=1e-10)->np.ndarray:
    """
    generates the candidates for non uniform embeddings
    can be used as uniform embeddings without any gaps

    Parameters
    ----------
    arrays : tuple
        what need to be embedded.
    max_lag : int
    noise_level : float, optional
        noise to add to prevent degeneracies. The default is 1e-10.

    Returns
    -------
    np.ndarray
        the candidates.

    """
    gapped = []
    A_shapes = []
    origin = []
    for k, A in enumerate(arrays):
        for order in range(1, max_lag):
            A_shapes.append(A.shape[1])
            origin.append(k)
            shortened = A[order:A.shape[0] - max_lag + order]
            shortened = np.ascontiguousarray(shortened)
            shortened = shortened.reshape(shortened.shape[0],-1)
            noise = noise_level*np.random.random_sample(shortened.shape)
            shortened = shortened + noise
            extended = np.zeros((shortened.shape[0]+2,shortened.shape[1]))
            extended[0] = k
            extended[1] = order
            extended[2:] = shortened
            gapped.append(extended)
    gapped = nbhstack(gapped)
    return gapped, np.array(A_shapes, dtype=np.int64)

@nb.njit([float64(float64[:,:],float64[:,:],float64[:,:],int64)],parallel=True)
def bootstrap_info_test(target:np.ndarray,
                        candidate:np.ndarray,
                        embedding:np.ndarray,
                        k:int)->float:
    """
    test for significant information transfer from target to candidate
    excluding that from the embeddings

    Parameters
    ----------
    target : np.ndarray
    candidate : np.ndarray
    embedding : np.ndarray
    k : int
        nearest neighbour parameter.

    Returns
    -------
    float
        99% significance value threshold for significance.

    """
    te_record = np.zeros(100)
    for i in nb.prange(100):
        shuffled_target = target[np.random.randint(0,target.shape[0],size=target.shape[0])]
        shuffled_candidate = candidate[np.random.randint(0,candidate.shape[0],size=candidate.shape[0])]
        te,err = KSG_TE_nb(shuffled_target,
                                 shuffled_candidate,
                                 embedding,
                                 k,
                                 local=False, 
                                 normalize=False)
        te_record[i] = te[0]
    return np.percentile(te_record,99)

def make_nue(target:np.ndarray,
              sources:tuple,
              dim:int = np.inf,
              max_lag:int = 5,
              blend:float = .75,
              thresh:float = 0,
              return_separated:bool = True,
              noise_level:float = 1e-10,
              use_bootstrap:bool = False,
              use_dk:bool=False) -> np.ndarray:
    """
    generates the non uniform embeddings for information based measures

    Parameters
    ----------
    target : np.ndarray
        the target array.
    sources : tuple
        the candidate reserves.
    dim : int, optional
        maximum embedding dimension. The default is np.inf.
    max_lag : int, optional
        DESCRIPTION. The default is 5.
    blend : float, optional
        blend of usage of information and regression error. The default is .75.
    thresh : float, optional
        information and error detection threshold. The default is 0.
    return_separated : bool, optional
        return as a dictionary of separated arrays or as one hot array. 
        The default is True.
    noise_level : float, optional
        noise to add to any synthetic samples to remove degeneracy. The default is 1e-10.
    use_bootstrap : bool, optional
        use bootstrap detection instead of MRS. The default is False.

    Returns
    -------
    TYPE
        the embeddings either as a dictionary with the index of source array as keys,
        or as an one hot array with everything in it
    """
    #we'll need to do this for each combination of the spaces
    assert len(target.shape) == 2, "incorrect shape"
    for source in sources:
        assert len(source.shape) == 2, "incorrect shape"
    space, cand_shapes = construct_candidates(sources,max_lag,noise_level)
    # we note that the space is constructed with time ordering in mind
    target = target[:space.shape[0]-2] #shorten the target if needed
    #we note that the NN_CMI returns the same k for similar data
    #even for joint and marginal spaces. so compute once, use later
    emb = np.array([[],],dtype = np.float64)
    _,k = NN_CMI(target, space[2:], emb,2,False,False)
    st,fn, spid = rank_by_MRS(target, space[2:], cand_shapes, emb, k, blend,use_dk)
    ranked = space[:,st:fn]#st upto not including fn
    # rank top k for the target
    emb=ranked
    space = np.hstack([space[:,:st], space[:,fn:]])
    cand_shapes = np.delete(cand_shapes, spid)
    #remap the shapes to the space
    while space.shape[1] > 0 and emb.shape[1] <= dim:
        # _,k = NN_CMI(target, space[2:], emb[2:],2,False,False)
        #untill they add chebyshev metric to the numba kd trees, this would be it
        st,fn, spid = rank_by_MRS(target,
                            space[2:],
                            cand_shapes,
                            emb[2:],
                            k,
                            blend,
                            use_dk)
        ranked = space[:,st:fn]
        U = np.hstack((emb, ranked))
        if use_bootstrap:
            rejection_limit = bootstrap_info_test(target,ranked[2:],emb[2:],k)
            current_te,err = KSG_TE_nb(target, ranked[2:], emb[2:],k,
                                       local=False,normalize=False)
            keep_on = current_te[0] > rejection_limit
        else:
            old_err = MRS(target, emb[2:], k, local=False)[0]
            new_err = MRS(target, U[2:], k, local=False)[0]
            keep_on = (old_err - new_err >= thresh)
        if keep_on:
            # print(f"current loop {j}")
            emb = U  # rotate back to being row-wise
            space = np.hstack([space[:,:st], space[:,fn:]])
            cand_shapes = np.delete(cand_shapes, spid)
        else:
            break  # stop if the best match is no longer valid
    uniques = np.unique(emb[0])
    emb_dict = {i:emb[2:,np.argwhere(emb[0]==i).flatten()] for i in uniques}
    if return_separated:
        return emb_dict
    else:
        return emb[2:]

def extend_sample(sample:np.ndarray,n_more:int)->np.ndarray:
    """
    #extends the array by n_more without dramatically changing statistics
    #we should note that sample should be 1D array (list like)

    Parameters
    ----------
    sample : np.ndarray
        original samples, each row is treated as a vector point.
    n_more : int
        how many more points.

    Returns
    -------
    np.ndarray
        the extended sample.

    """
    
    generator = np.random.default_rng()
    sample_size = len(sample)
    bayesian =  sample_size < 100  
    if bayesian:
        #bayesian bootstrapping
        al = [4 for _ in range(sample_size)]
        weights = generator.dirichlet(al,n_more)
        idxs = [generator.choice(sample_size,p=weight) for weight in weights]
        extras = sample[idxs]
    else:
        #regular old bootstrap
        extras = generator.choice(sample,n_more)
    return np.vstack((sample,extras))


def make_ue(A, order, gap):
    emb = np.vstack([np.hstack(A[i-order*gap:i:gap]) for i in range(order*gap,len(A))])
    emb = np.array(emb)
    emb = extend_sample(emb, len(A)-len(emb))
    return emb

@nb.njit(parallel=True)
def row_wise_intersection(A:np.ndarray,B:np.ndarray)->np.ndarray:
    """
    finds the elements of A in each row that are also in B
    might be needed for DK_TE_ if there are too many rows
    
    Parameters
    ----------
    A : np.ndarray
        The array with the smallest width, so shortest rows per each column
        it's ok to use the bigger array as A, just inefficent
    B : np.ndarray
        The intersecting array, where each row may intersect with rows of A
    Returns
    -------
    C : np.ndarray
        The intersection, initialized as an array of -1s with same shape as A
    """
    #might use this for counting
    assert A.shape[0] == B.shape[0], "different length arrays"
    C = np.full(A.shape,-1)
    for i in nb.prange(A.shape[0]):
        a = A[i][A[i]!=-1] #this is not availiable
        b = B[i][B[i]!=-1]
        c = np.full(a.shape,-1)
        for j in range(len(a)):
            if a[j] in b:
                c[j]=a[j]
        c.sort()
        C[i,:c.shape[0]] = np.flip(c)
    end = bt.find_column(C,-1)
    if end is not None:
        C = C[:,:end]
    return C

@nb.njit(parallel=True)
def count_per_column(A:np.ndarray,value:float=-1)->np.ndarray:
    """
    

    Parameters
    ----------
    A : np.ndarray
        array to be counted for each row, highest count is the column count
    value : float, optional
        value to exclude from array. The default is -1.

    Returns
    -------
    counts : np.ndarray
        the count of each row that isn't the value specified.

    """
    counts = np.zeros(A.shape[0])
    for i in nb.prange(A.shape[0]):
        a = A[i][A[i] != value]
        counts[i] = len(a)
    return counts

@nb.njit
def DK_TE_(Yt:np.ndarray,
           X:np.ndarray,
           Y:np.ndarray,
           h:int,
           local:bool = False)->float:
    """
    the density kernel method to find transfer entropy, not optimized yet

    Parameters
    ----------
    Yt : np.ndarray
        target array.
    X : np.ndarray
        source array.
    Y : np.ndarray
        target history or information to exclude.
    h : int
        smoothing kernel parameter.
    local : bool, optional
        local entropies. The default is False.

    Returns
    -------
    float
        entropy.

    """
    assert Yt.shape[0] == Y.shape[0] == X.shape[0], "inhomogenous lengths"
    
    #construct KDtree in each space and find kth neighbout
    yt = bt.iterative_construct(Yt, metric="chebyshev")
    y = bt.iterative_construct(Y, metric="chebyshev")
    x = bt.iterative_construct(X, metric="chebyshev")
    
    #distances separetly in each space, by requiring h points in each ball
    yid, yd = bt.query(Y,Y,y,k=h+1)
    ytid, ytd = bt.query(Yt,Yt,yt,k=h+1)
    xid, xd = bt.query(X,X,x,k=h+1)
    #construct constraints on joint spaces
    xid, xd = bt.query_radius(X,X,x,xd[:,h],max_count=h+1,count_only=False)
    yid, yd = bt.query_radius(Y,Y,y,yd[:,h],max_count=h+1,count_only=False)
    ytid, ytd = bt.query_radius(Yt,Yt,yt,ytd[:,h],max_count=h+1,count_only=False)
    
    #we note that the id and d are sorted by distance
    if xid.shape[1] < yid.shape[1]:
        xy = row_wise_intersection(xid,yid)
        end = bt.find_column(xy,-1)
        if end is not None:
            xy = xy[:,:end]
    else:
        xy = row_wise_intersection(yid,xid)
        end = bt.find_column(xy,-1)
        if end is not None:
            xy = xy[:,:end]
    if ytid.shape[1] < yid.shape[1]:
        yty = row_wise_intersection(ytid,yid)
        end = bt.find_column(yty,-1)
        if end is not None:
            yty = yty[:,:end]
    else:
        yty = row_wise_intersection(yid,ytid)
        end = bt.find_column(yty,-1)
        if end is not None:
            yty = yty[:,:end]
    if yty.shape[1] < xy.shape[1]:
        ytxy = row_wise_intersection(yty,xy)
        end = bt.find_column(ytxy,-1)
        if end is not None:
            ytxy = ytxy[:,:end]
    else:
        ytxy = row_wise_intersection(xy,yty)
        end = bt.find_column(ytxy,-1)
        if end is not None:
            ytxy = ytxy[:,:end]
    hxy = count_per_column(xy,-1)
    hyty = count_per_column(yty,-1)
    hytxy = count_per_column(ytxy,-1)
    #the some is over time, multiplication is done each slice
    base = lambda r: np.log2((r*h)/(hyty*hxy))
    TEb = np.zeros(hxy.shape)
    for r in range(1,h+1):
        # Pb = hypergeom.pmf(n = hyty-1, k = r-1, M=h-1, N=hxy-1)
        Pb = vbinom(hyty-1,float(r-1)) * vbinom(h-hyty,hxy-r) / vbinom(h-1,hxy-1)
        Pb = np.nan_to_num(Pb)
        TEb = TEb + Pb * np.nan_to_num(base(r))
    #we may notice that pb isn't normalized, so lets normalize
    if local:
        TE0 = np.nan_to_num(base(hytxy))
        TE = TE0 - TEb
        return TE
    else:
        TE0 = np.mean(np.nan_to_num(base(hytxy)))
        #we might need to vectorize
        TEb = np.mean(TEb)
        TE = TE0 - TEb
        return np.array([TE])

# @timeit
@nb.njit
def DK_TE(Yt:np.ndarray,X:np.ndarray,Y:np.ndarray,local:bool=False)->float:
    """
    the density kernel method to find transfer entropy, optimized.

    Parameters
    ----------
    Yt : np.ndarray
        target array.
    X : np.ndarray
        source array.
    Y : np.ndarray
        target history or information to exclude.
    local : bool, optional
        local entropies. The default is False.

    Returns
    -------
    float
        entropy.

    """
    N,my = X.shape
    guess_k = int(np.ceil(N**.5))
    a = guess_k/2
    b = guess_k*1.5
    invphi = (5**.5 - 1)/2
    tolerance = 1/invphi + .5
    while b - a > tolerance:
        c = int(np.floor(b - (b - a) * invphi)) #new lower bound
        d = int(np.ceil(a + (b - a) * invphi)) #new upper bound
        Yt = np.ascontiguousarray(Yt)
        X = np.ascontiguousarray(X)
        Y = np.ascontiguousarray(Y)
        fc = DK_TE_(Yt,X,Y,h=c,local=False)
        fd = DK_TE_(Yt,X,Y,h=d,local=False)
        if fc[0] > fd[0]: #the function to the lower bound side is smaller
            b = d #elimiate the side to the upper bound
        else:
            a = c #elimiate the other side
    k_opm = int(np.ceil((b + a) / 2)) #take the middle
    I = DK_TE_(Yt,X,Y,h=k_opm,local=local)
    return I


def CCA_TE(Yt:np.ndarray,X:np.ndarray, Y:np.ndarray)->float:
    """
    CCA method of finding TE of gaussian distributed variables
    TE = I(Yt:X,Y) - I(Yt:Y)
    
    Parameters
    ----------
    Yt : np.ndarray
        target array.
    X : np.ndarray
        source array.
    Y : np.ndarray
        target history or information to exclude.

    Returns
    -------
    float
        transfer entropy.

    """
    XY = np.hstack((X,Y))
    cmp = min(X.shape[-1],Y.shape[-1],XY.shape[-1],Yt.shape[-1])
    cca = CCA(n_components=cmp)
    cca.fit(Yt,XY)
    ytc,xyc = cca.transform(Yt,XY)
    cor_ytxy = np.corrcoef(ytc.flatten(),xyc.flatten())[0,1]
    Iytxy = np.sum(-np.log2(np.sqrt(1-cor_ytxy**2))/2)
    
    cca = CCA(n_components=cmp)
    cca.fit(Yt,Y)
    ytc,yc = cca.transform(Yt,Y)
    cor_yty = np.corrcoef(ytc.flatten(),yc.flatten())[0,1]
    Iyty = np.sum(-np.log2(np.sqrt(1-cor_yty**2))/2)
    TE = Iytxy - Iyty
    return TE

    
    