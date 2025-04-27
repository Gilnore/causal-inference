# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 12:02:09 2025

@author: robin
"""

from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
from time import perf_counter
import numpy as np
from ball_tree import iterative_construct, query, query_radius
import numba as nb

def plot_leaves(tree, data):
    #suppose we have a tree of 2d points
    assert data.shape[1] == 2, "data dimension too high or low to plot, use 2D points."
    fig, ax = plt.subplots()
    # circles = []
    centers, radii, children, leaves, metric = tree
    for leaf in leaves:
        node = int(leaf[-1])
        circle = plt.Circle(xy=centers[node],radius=radii[node])
        # circles.append(circle)
        ax.add_artist(circle)
        
        circle.set_facecolor("none")
        # circle.set_clip_box(ax.bbox)
        circle.set_edgecolor("red")
        circle.set_alpha(1)
    ax.scatter(data[:,0],data[:,1])
    plt.show()

def validate_leaves(leaves,n=100):
    leaves = leaves[:,:-1]
    flat_leaves = leaves.flatten()
    flat_leaves = np.array(sorted(flat_leaves))
    flat_leaves = flat_leaves[flat_leaves!=-1]
    if len(flat_leaves) == n:
        return all(flat_leaves==np.arange(n))
    else: return False


@nb.njit
def test_tree(data):
    tree = iterative_construct(data, metric="chebyshev")
    inds,dists = query(data, data, tree, 10)
    r = np.full(data.shape[0],1)
    count, dists_r = query_radius(data, data, tree, r,
                            count_only=True,
                           max_count=100
                          )
    return inds, dists, count

if __name__ == "__main__":
    # nb.config.DISABLE_JIT = True
    my_queries = []
    sk_queries = []
    idx_correct = []
    dist_correct = []
    count_correct = []
    for i in range(1,20):
        n = 100*i
        data =np.random.randn(n,8)
        t = perf_counter()
         
        t0 = perf_counter()
        inds, dists, count = test_tree(data)
        t1 = perf_counter()
        ref_tree = BallTree(data, metric="chebyshev")
        # ref_tree = KDTree(data,leafsize=40)
        
        dists_ref,inds_ref = ref_tree.query(data,k=10)
        count_ref = ref_tree.query_radius(data,1,
                                           count_only=True)
        t2 = perf_counter()
        mine= t1-t0
        sk = t2-t1
        my_queries.append(mine)
        sk_queries.append(sk)
        idx_correct.append(np.all(inds == inds_ref))
        dist_correct.append(np.all(dists == dists_ref))
        count_correct.append(np.all((count_ref - count.flatten())==0))

    plt.figure("queries")
    plt.plot(my_queries[1:], label = "mine")
    plt.plot(sk_queries[1:], label = "sklearn")
    plt.legend()
    plt.show()

    print("correct idx", all(idx_correct))
    print("correct dists", all(dist_correct) )
    print("correct count", all(count_correct))
    