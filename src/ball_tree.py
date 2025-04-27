# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 10:49:40 2025

@author: robin

"""
from numba import float64#, int64
import numpy as np
import numba as nb
from typing import Tuple
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
from time import perf_counter

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


@nb.njit(fastmath=True)
def dist(source:np.ndarray,target:np.ndarray,metric:str="euclidean")->float:
    """
    #this is the single variable metric function for a vector source and a vector target
    #both source and target are 1D array row vector

    Parameters
    ----------
    source : np.ndarray
        from this vector.
    target : np.ndarray
        to this vector.
    metric : str, optional
        metric type. The default is "euclidean", can be chebyshev

    Returns
    -------
    float
        distance.

    """
    
    if metric == "euclidean":
        dif = (source - target)
        res = np.dot(dif,dif)**.5
        return res
    elif metric == "chebyshev":
        return np.max(vabs(source - target))

@nb.njit(fastmath=True)
def split(data,metric:str="euclidean")->Tuple[float,float]:
    """
    split the array on a random seed

    Parameters
    ----------
    data : TYPE
        points to split.
    metric : str, optional
        metric type. The default is "euclidean", can be chebyshev

    Returns
    -------
    Tuple[float,float]
        left furthest point from seed and right after split.

    """
    length = len(data)
    np.random.seed(42)
    seed = np.random.randint(0,length)
    point = data[seed]
    #next we find furthest point to seed
    xl = 0
    xld = 0
    for i in nb.prange(length):
        dists_i = dist(point,data[i],metric)
        if dists_i > xld:
            xl = i
            xld = dists_i
    #next we find furthest point to furthest point to seed
    xr = 0
    xrd = 0
    for i in nb.prange(length):
        dists_i = dist(data[xl],data[i],metric)
        if dists_i > xrd:
            xrd = dists_i
            xr = i
    #return the pair of indicies
    if xld == xrd == 0:
        #this is just a single point
        data += np.random.random_sample(data.shape)*1e-10
        xl,xr = split(data,metric)
    return (xl,xr)

@nb.njit(fastmath=True)
def find_cent_rad(subset:np.ndarray,metric:str)->Tuple[np.ndarray,float]:
    """
    find center and radius of a point set

    Parameters
    ----------
    subset : np.ndarray
    metric : str
        metric type. The default is "euclidean", can be chebyshev

    Returns
    -------
    np.ndarray 1d
        center point vector.
    float
        radius of set from center.

    """
    subset_length = subset.shape[0]
    if subset_length == 1:
        return subset[0], 0
    # Compute center manually (avoiding np.mean for Numba compatibility)
    else:
        Nc = np.zeros(subset.shape[1])
        for i in nb.prange(subset_length):
            Nc += subset[i]
        Nc /= subset_length  # Compute mean
        # Compute max radius (distance from center to farthest point)
        Nr = 0
        for i in nb.prange(subset_length):
            dist_i = dist(Nc, subset[i], metric)
            if dist_i > Nr:
                Nr = dist_i
        return Nc,Nr

@nb.njit(fastmath=True)
def iterative_construct(dataset: np.ndarray, 
                        leaf_size: int = 40, 
                        metric:str="euclidean")->tuple:
    """
    Iteratively construct a KD-tree without recursion.
    
    Parameters:
        dataset (np.ndarray): Input dataset as a 2D array where each row is a point.
        N0 (int): Minimum number of points to stop splitting and form a leaf.
        metric: distance metric: "euclidean" or "chebyshev"
    Returns:
        tuple: (centers, radii, children, leaves) representing the tree structure.
        note on leaf, it stores the node it stems from in the last index,
        other indices of the leaf stores indicies of the dataset
    """
    N0 = leaf_size
    stack = []  # Stack for iterative traversal
    centers = []  # List to store node centers
    radii = []  # List to store node radii
    children = []  # List to store child node indices (-1 for leaves)
    leaves = np.zeros((1,N0+1),dtype=np.int64) - 1 #default index is -1 
    #not parallel, contain (indeices of leaf, index of center it belong to)
    initial_itter = True
    # Push root node onto stack
    indexed_dataset = np.zeros((dataset.shape[0],dataset.shape[1]+1))
    indexed_dataset[:,:-1] = dataset
    indexed_dataset[:,-1] = np.arange(dataset.shape[0]) #store index with dataset
    stack.append((indexed_dataset, -1, 0)) 
    #data, parent indes, is left child, data index

    while stack: #while stack is not empty
        indexed_subset, parent_idx, is_left = stack.pop()
        subset = indexed_subset[:,:-1] 
        subset_length = len(subset)
        #we'll use subset in calculations, and pass on indexed version
        Nc,Nr = find_cent_rad(subset,metric)
        if Nr <= 1e-11 and subset_length > 1:
            noise = np.random.random_sample(subset.shape)
            subset += noise* 1e-10
            Nc, Nr = find_cent_rad(subset, metric) 
        # Store node information
        node_index = len(centers) #grows as loop iterates, keeps list parallel
        centers.append(Nc)
        radii.append(Nr)
        children.append((-1, -1, parent_idx))  # Default: No children

        # Update parent reference from the default
        if parent_idx != -1:
            left, right, parent = children[parent_idx] #left and right of parent node
            if is_left:
                children[parent_idx] = (node_index, right, parent_idx) 
                #all these are idxs in tree, since we might need back tracking add parent
            else:
                children[parent_idx] = (left, node_index, parent_idx)

        # If subset is larger than threshold, continue splitting
        if subset_length > N0:
            xl, xr = split(subset,metric)  # Find two farthest points
            left_cent = subset[xl]
            right_cent = subset[xr]

            left_children = []
            right_children = []

            # Partition points into left and right subsets
            for i in range(subset_length):
                point = subset[i]
                if dist(point, left_cent,metric) <= dist(point, right_cent,metric):
                    left_children.append(indexed_subset[i])
                else:
                    right_children.append(indexed_subset[i])

            # Convert lists to NumPy arrays in a Numba-friendly way
            if len(left_children) > 0:
                left_children_np = np.empty((len(left_children),
                                             indexed_subset.shape[1]),
                                            dtype=np.float64)
                for i in range(len(left_children)):
                    left_children_np[i] = left_children[i]
                stack.append((left_children_np, node_index, 1)) 
                #the last index is left

            if len(right_children) > 0:
                right_children_np = np.empty((len(right_children),
                                              indexed_subset.shape[1]),
                                             dtype=np.float64)
                for i in range(len(right_children)):
                    right_children_np[i] = right_children[i]
                stack.append((right_children_np, node_index, 0))
        else:#these points are leaf points
            #find the indexes of in subset points
            leaf = np.zeros(N0+1,dtype=np.int64) - 1
            leaf_inds = np.zeros(subset_length,dtype=np.int64)
            for p in range(subset_length):
                leaf_inds[p] = indexed_subset[p,-1]
            for l in range(len(leaf_inds)):
                leaf[l] = leaf_inds[l]
            leaf[-1] = node_index 
            #we stored the node index instead of the dataset index
            if initial_itter:
                leaves[0] = leaf
            else:
                new_leaves = np.zeros((leaves.shape[0]+1,N0+1),dtype=np.int64)-1
                for l in range(leaves.shape[0]):
                    new_leaves[l] = leaves[l]
                new_leaves[-1] = leaf
                leaves = new_leaves
            initial_itter=False
    
    np_centers = np.zeros((len(centers), centers[0].shape[0]))
    for i in range(np_centers.shape[0]):
        np_centers[i] = centers[i]
    centers = np_centers
    np_radii = np.zeros(len(radii))
    for i in range(np_radii.shape[0]):
        np_radii[i] = radii[i]
    radii = np_radii
    np_children = np.zeros((len(children),3),dtype=np.int64)
    for i in range(np_children.shape[0]):
        np_children[i][0],np_children[i][1],np_children[i][2] = children[i]
    children = np_children
    #we need to convert this to tuple of arrays and str
    return (centers, radii, children, leaves, metric)

@nb.njit(fastmath=True, parallel=True)
def query(points:np.ndarray, 
          data:np.ndarray,
          tree:Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,str],
          k:int)->Tuple[np.ndarray,np.ndarray]:
    #initiate the return arrys
    inds = np.zeros((points.shape[0],k), dtype=np.int64)
    dists = np.zeros((points.shape[0],k))
    centers, radii, children, leaves, metric = tree
    #we'll switch to using prange later
    for p in nb.prange(points.shape[0]):
        stack = [0] #make the query stack
        neighbour_candidates = np.zeros((k,2))-1 #(ind,dist) pair, sort later
        neighbour_candidates[:,1] = np.inf #initiate closest distance at infinity
        while stack:
            #we start at 0th position in all parallel arrays
            pos = stack.pop() #this is a tree index
            left,right,parent = children[pos]
            rad = radii[pos]
            cent = centers[pos]
            point = points[p]
            out_dist = dist(point,cent,metric) - rad
            if out_dist < neighbour_candidates[-1,1]:
                is_leaf = left == right == -1
                #the current node is a better guess than at least one of the neigbours
                if is_leaf:
                    leaf_loc = np.argwhere(leaves[:,-1]==pos)[0,0]
                    leaf_idx = leaves[leaf_loc,:-1][leaves[leaf_loc,:-1]!=-1]
                    for i in leaf_idx:
                        pt = data[i]
                        pt_dist = dist(pt,point,metric)
                        if pt_dist < neighbour_candidates[-1,1]:
                            #replace the furthest candidate
                            neighbour_candidates[-1,0] = i
                            neighbour_candidates[-1,1] = pt_dist
                            sort_per = np.argsort(neighbour_candidates[:,1])
                            neighbour_candidates = neighbour_candidates[sort_per]
                else:
                    #internal node
                    cl = centers[left]
                    cr = centers[right]
                    dcl = dist(point, cl,metric)
                    dcr = dist(point, cr,metric)
                    if dcl < dcr:
                        stack.append(left)
                        stack.append(right)
                    else:
                        stack.append(right)
                        stack.append(left)

        #update the record for that point
        inds[p] = neighbour_candidates[:,0]
        dists[p] = neighbour_candidates[:,1]
    return inds, dists

@nb.njit
def find_column(A:np.ndarray, value:float = -1)->int:
    """
    find first column filled with value in A

    Parameters
    ----------
    A : np.ndarray
    value : float, optional
        The default is -1.

    Returns
    -------
    int
        idx.

    """
    for i in range(A.shape[1]):
        if np.all(A[:,i]==value):
            return i

@nb.njit(fastmath=True, parallel=True)
def query_radius(points:np.ndarray, 
                data:np.ndarray,
                tree:Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,str],
                r:np.ndarray,
                max_count:int = np.inf,
                count_only:bool = False)->Tuple[np.ndarray,np.ndarray]:
    if count_only:
        inds = np.zeros((points.shape[0],1), dtype=np.int64)
        dists = np.zeros((points.shape[0],1))
    else:
        k = int(min(max_count, points.shape[0])) + 1
        #initiate the return arrys
        inds = np.zeros((points.shape[0],k), dtype=np.int64)
        dists = np.zeros((points.shape[0],k))
    centers, radii, children, leaves, metric = tree
    multi_r = r.shape[0] > 1
    #we'll switch to using prange later
    for p in nb.prange(points.shape[0]):
        if multi_r:
            r_p = r[p] #radius required by point
        else:
            r_p = r[0]
        stack = [0] #make the query stack
        if count_only:
            neighbour_candidates = np.zeros((1,2))
        else:
            neighbour_candidates = np.zeros((k,2))-1 #(ind,dist) pair, sort later
            neighbour_candidates[:,1] = np.inf #initiate closest distance at infinity
        while stack:
            #we start at 0th position in all parallel arrays
            pos = stack.pop() #this is a tree index
            left,right,parent = children[pos]
            rad = radii[pos] #radius of ball
            cent = centers[pos]
            point = points[p]
            out_dist = dist(cent , point, metric) - rad#point to center of sphere
            if out_dist <= r_p:
                is_leaf = left == right == -1
                #the current node is a better guess than at least one of the neigbours
                if is_leaf:
                    leaf_loc = np.argwhere(leaves[:,-1]==pos)[0,0]
                    leaf_idx = leaves[leaf_loc,:-1][leaves[leaf_loc,:-1]!=-1]
                    for i in leaf_idx:
                        pt = data[i] #leaf point
                        pt_dist = dist(pt, point, metric)
                        if count_only and pt_dist <= r_p:
                            neighbour_candidates[0,0] += 1
                        elif pt_dist < neighbour_candidates[-1,1] and pt_dist <= r_p:
                            #replace the furthest candidate
                            neighbour_candidates[-1,0] = i
                            neighbour_candidates[-1,1] = pt_dist
                            sort_per = np.argsort(neighbour_candidates[:,1])
                            neighbour_candidates = neighbour_candidates[sort_per]
                else:
                    #internal node
                    cl = centers[left]
                    cr = centers[right]
                    dcl = dist(point, cl, metric)
                    dcr = dist(point, cr, metric)
                    if dcl < dcr:
                        stack.append(left)
                        stack.append(right)
                    else:
                        stack.append(right)
                        stack.append(left)
                        
        #update the record for that point
        if count_only:
            inds[p] = neighbour_candidates[0,0]
        else:
            inds[p] = neighbour_candidates[:,0]
            dists[p] = neighbour_candidates[:,1]
    if not count_only:
        end = find_column(inds,-1)
        if end is not None:
            inds = inds[:,:end]
            dists = dists[:,:end]
    return inds, dists

