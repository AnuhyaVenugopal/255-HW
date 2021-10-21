#!/usr/bin/env python
# coding: utf-8

# In[74]:


import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from collections import defaultdict
from sklearn.utils import shuffle
from sklearn.metrics import calinski_harabasz_score
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[75]:


with open("train.dat", "r", encoding="utf8") as f:
    lines = f.readlines()
docs = [l.split() for l in lines]
def filterLen(docs, minlen):
    return [ [t for t in d if len(t) >= minlen ] for d in docs ]
docs1 = filterLen(docs, 4)
from collections import Counter
from scipy.sparse import csr_matrix
def build_matrix(docs):
    nrows = len(docs)
    idx = {}
    tid = 0
    nnz = 0
    for d in docs:
        nnz += len(set(d))
        for w in d:
            if w not in idx:
                idx[w] = tid
                tid += 1
    ncols = len(idx)
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0  
    n = 0  
    for d in docs:
        cnt = Counter(d)
        keys = list(k for k,_ in cnt.most_common())
        l = len(keys)
        for j,k in enumerate(keys):
            ind[j+n] = idx[k]
            val[j+n] = cnt[k]
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1        
    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()
    return mat
def csr_info(mat, name="", non_empy=False):
    if non_empy:
        print("%s [nrows %d (%d non-empty), ncols %d (%d non-empty), nnz %d]" % (
                name, mat.shape[0], 
                sum(1 if mat.indptr[i+1] > mat.indptr[i] else 0 
                for i in range(mat.shape[0])), 
                mat.shape[1], len(np.unique(mat.indices)), 
                len(mat.data)))
    else:
        print( "%s [nrows %d, ncols %d, nnz %d]" % (name, 
                mat.shape[0], mat.shape[1], len(mat.data)) )
mat = build_matrix(docs)
mat1 = build_matrix(docs1)
csr_info(mat)
csr_info(mat1)


from collections import defaultdict
def csr_idf(mat, copy=False, **kargs):
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    df = defaultdict(int)
    for i in ind:
        df[i] += 1
    for k,v in df.items():
        df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory
    for i in range(0, nnz):
        val[i] *= df[ind[i]]        
    return df if copy is False else mat
def csr_l2normalize(mat, copy=False, **kargs):
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0/np.sqrt(rsum)
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum      
    if copy is True:
        return mat
mat2 = csr_idf(mat1, copy=True)
mat3 = csr_l2normalize(mat2, copy=True)
cmat=mat3.toarray()


# In[76]:


from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=500)
X = svd.fit_transform(cmat)


# In[77]:


from sklearn.metrics import pairwise_distances
def getCentroid(matrix, clusters):
    centroids = []
    for i in range(0,2):
        cluster = matrix[clusters[i],:]
        centroids.append(cluster.mean(0))
    return np.asarray(centroids)


# In[78]:


def kmeans(matrix, n):
    shuffledMatrix = shuffle(matrix, random_state=0)
    centroids = shuffledMatrix[:2,:]
    for _ in range(n):
        c = []
        c1 = []
        c2 = []
        sim =  matrix.dot(centroids.T)
        for index in range(sim.shape[0]):
            similarityRow = sim[index]
            sortedSimilarity = np.argsort(similarityRow)[-1]
            if sortedSimilarity == 0:
                c1.append(index)
            else:
                c2.append(index)
        if len(c1) > 1:
            c.append(c1)
        if len(c2) > 1:
            c.append(c2)
        centroids = getCentroid(matrix, c)  
    return c1, c2


# In[79]:


#implementing bisecting kmeans
def bkm(input, k, n):
    #setting the cluster
    data = []
    temp = []
    for i in range(input.shape[0]):
        temp.append(i)
    data.append(temp)
    while len(data) < k:
        sse = []
        for cluster in data:
            members = input[cluster,:]
            sse.append(np.sum(np.square(members - np.mean(members))))   
        di = np.argsort(np.asarray(sse))[-1]
        dc = data[di]
        c1, c2 = kmeans(input[dc,:], n)
        del data[di]
        first = []
        second = []
        for i in c1:
            first.append(dc[i])
        for i in c2:
            second.append(dc[i])
        data.append(first)
        data.append(second)
    out = [0] * input.shape[0]
    for i, cluster in enumerate(data):
        for j in cluster:
            out[j] = i + 1
    return out


# In[80]:


kValues = list()
scores = list()
for k in range(3, 22, 2):
    labels = bkm(X, k, 10)
    if (k == 7):
        outputFile = open("anu.dat", "w")
        for index in labels:
            outputFile.write(str(index) +'\n')
        outputFile.close()
        
    score = calinski_harabasz_score(X, labels)
    kValues.append(k)
    scores.append(score)
plt.plot(kValues, scores)
plt.xticks(kValues, kValues)
plt.xlabel('Number of Clusters k')
plt.ylabel('Calinski and Harabaz Score')


# In[ ]:




