import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from zlib import adler32

"""
1) K-core decomposition
"""
def lay_offset(layout, offset = 0.01):
    new_lay = layout
    for node in layout:
        new_lay[node] = layout[node] + [0,offset]
    return new_lay

def k_core_decompose(G):
    k_core_decompose_ = nx.core_number(G)
    k_core_decompose_ = np.array(list(k_core_decompose_.values()))
    return k_core_decompose_

def k_core_decompose_fig(G,pos, labels = None):
    plt.figure(figsize=(4.5*2, 4.5*4.5))
    x_max, y_max = np.array(list(pos.values())).max(axis=0)
    x_min, y_min = np.array(list(pos.values())).min(axis=0)

    for i in range(8):
        plt.subplot(4, 2, i+1)
        subG = nx.k_core(G, i+1)
        nodes = nx.draw_networkx_nodes(
            subG,
            pos,
            cmap=plt.cm.OrRd,
            node_color=k_core_decompose(subG),
            node_size=100,
            edgecolors='black'
        )
        nx.draw_networkx_edges(
            subG,
            pos,
            alpha=0.3,
            width=1,
            edge_color='black'
        )
#         labels = [(name.split(' ')[0]  + ' ' + name.split(' ')[-1][:1] + '.') for name in list(nx.nodes(subG))]
#         labels = dict(zip(list(subG),labels))
        if labels is not None:
            nx.draw_networkx_labels(G, lay_offset(pos, offset = 0.02), verticalalignment = 'bottom', 
                                labels = labels, font_size = 7)
        eps = (x_max - x_min) * 0.05
        plt.xlim(x_min-eps, x_max+eps)
        plt.ylim(y_min-eps, y_max+eps)
        plt.legend(prop = {'size':5}, *nodes.legend_elements())
        plt.axis('off')
        plt.title('k-shells on {}-core'.format(i+1))
    plt.tight_layout()

"""
2) Clique detection
"""
def largest_cliques(G):
    cliq = list(nx.find_cliques(G))

    max_cliqs = [i for i in cliq if len(i) == len(max(cliq, key=len))]

    nod = []

    for i in max_cliqs:
        cli = []

        for j in G.nodes:

            if j in i:
                cli.append([0.1, 0.3, 0.5])
            else:
                cli.append([1, 1, 1])

        nod.append(cli)

    nod_array = np.array(nod)

    wid = []

    cl_ed = [[(j, k) for j in i for k in i] for i in max_cliqs]

    for k in cl_ed:

        cli_w = []

        for e in G.edges:

            Gi = G.subgraph(k).copy()

            if e in k:

                cli_w.append(1)

            else:

                cli_w.append(0.5)

        wid.append(cli_w)

    weed = np.array(wid)

    return nod_array, weed

def cliques_fig(G,pos,colors,widths,labels):
    plt.figure(figsize=(9,5))
    size = np.unique(colors[0], axis=0, return_counts=True)[1][0]
    for i in range(colors.shape[0]):
        b_edges = np.array(list(G.edges))[widths[i] == widths[i].max()]

        plt.subplot(colors.shape[0]//2,2, i+1)
        l = {0:'1',1:'2',2:'3',3:'4',4:'6'}
        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            node_color=colors[i],
            node_size=100,
            linewidths=1,
            edgecolors='black',
        )
        nx.draw_networkx_edges(
            G,
            pos,
            alpha=0.2,
            width=widths[i].min()
        )
        nx.draw_networkx_edges(
            G,
            pos,
            width=widths[i].max(),
            edgelist=b_edges
        )
        mask = [False if c.sum() == 3  else True for c in colors[i]]
        masked_names = dict(np.array(list(labels.items()))[mask])
        node_labels = nx.draw_networkx_labels(G, pos, verticalalignment = 'bottom', 
                                labels = masked_names, font_size = 7)
        legend = ''
        for name in masked_names.values():
            legend+=name+'\n'
        plt.legend([legend], prop = {'size':6}, loc = 'upper left', title = 'Nodes in clique')
        plt.title('Clique of the size {}'.format(size))
        plt.axis('off')
        plt.show()

"""
3) Modularity
"""
def modularities(G, n):
    c = nx.algorithms.community.girvan_newman(G)
    c_n = [i for i in c if len(list(i)) <= n + 1]
    return np.array([nx.algorithms.community.modularity(G, i) for i in c_n])

def modularities_fig(modularity, n_iterations):
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(n_iterations) + 2, modularity)
    best_n = np.argmax(modularity) + 2
    label = 'number of communities with max modularity {:.2f}'.format(max(modularity))
    plt.plot(
        [best_n, best_n], [min(modularity), max(modularity)],
        'k--', c='tab:red',
        label=label
    )
    plt.ylabel('Modularity score')
    plt.xlabel('Number of communities')
    plt.legend(loc='upper left')
    plt.ylim((modularity.min(), 0.6))
    plt.xticks(range(0,n_iterations,5))
    plt.show()

"""
4 Laplacian eigenmaps
"""
def Laplacian_eigenmaps(G, n_clusters, n_components):
    from numpy.linalg import inv
    def norm_laplacian(A):
        D = np.diag(sum(A))
        Di = (inv(D)) ** (1 / 2)
        Ln = Di @ (D - A) @ Di
        return Ln

    def spectral_embedding(L, n_components):
        _, eigvec = np.linalg.eigh(L)
        return (eigvec[:, 1:n_components + 1])

    A = nx.to_numpy_array(G)
    L = norm_laplacian(A)
    embedding = spectral_embedding(L, n_components)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embedding)
    return kmeans.labels_

def Laplacian_eigenmaps_fig(G,pos,n_clusters, n_components, labels):
    plt.figure(figsize=(8, 8))
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        cmap=plt.cm.rainbow,
        node_color= Laplacian_eigenmaps(G, n_clusters,  n_components),
        node_size=100,
        linewidths=1,
        edgecolors='black'
    )
    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.2,
        edge_color='black'
    )
    if labels is not None:
        nx.draw_networkx_labels(G, lay_offset(pos, offset = 0.02), verticalalignment = 'bottom', labels = labels, font_size = 7)
    plt.axis('off')
    plt.legend(*nodes.legend_elements())
    plt.show()
    plt.tight_layout()