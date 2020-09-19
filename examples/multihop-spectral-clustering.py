import random as rd
import numpy as np
import networkx as nx
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import metrics
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering


DG= nx.DiGraph()
G=nx.Graph()
Cluster=[14,15,16,17,18,19,20,21,22,26,122,125,56,57,58,64,65,67,68,71,98,47,49,50,59,61,93,94,42,7,8,9,10,12,13,23,24,25,28,29,30,31,33,34,35,36,37,38,39,40,41,43,44,48,60,90,123,62,63,92,95,99]

'''
Compute purity for clustering
'''
def purity(y_true,y_pred):
    contigency_matrix=metrics.cluster.contingency_matrix(y_true,y_pred)
    return np.sum(np.amax(contigency_matrix,axis=0))/np.sum(contigency_matrix)


'''
compute precision, recall and F score
for clustering
'''
def precision(y_true,y_pred):
    tp=0.0
    fp=0.0
    n1=len(y_true)
    for i in range(n1):
        for j in range(i+1,n1):
            if y_true[i]==y_true[j] and y_pred[i]==y_pred[j]:
                tp+=1
            if y_true[i]!=y_true[j] and y_pred[i]==y_pred[j]:
                fp+=1
    if tp==0 and fp==0:
        return 0
    else:
        return tp/(tp+fp)

def recall(y_true,y_pred):
    tp=0.0
    fn=0.0
    n1=len(y_true)
    for i in range(n1):
        for j in range(i+1,n1):
            if y_true[i]==y_true[j] and y_pred[i]==y_pred[j]:
                tp+=1
            if y_true[i]==y_true[j] and y_pred[i]!=y_pred[j]:
                fn+=1
    if tp==0 and fn==0:
        return 0
    else:
        return tp/(tp+fn)

def F_score(y_true,y_pred,beta):
    P=precision(y_true,y_pred)
    R=recall(y_true,y_pred)
    if P==0 and R==0:
        return 0
    else:
        return ((beta*beta+1)*P*R)/(beta*beta*P+R)



'''
test whether 3 nodes form motif instance of M6
'''
def is_motif(i,j,k):
    global DG
    nodes=[i,j,k]
    H=DG.subgraph(nodes)
    M6=nx.DiGraph()
    for u in range(3):
        M6.add_node(u)
    M6.add_edge(0,1)
    M6.add_edge(0,2)
    M6.add_edge(1,2)
    M6.add_edge(2,1)
    return nx.is_isomorphic(H,M6)
    


'''
compute total motif instances of M6
'''
def count_motif(i,j):
    global G
    nb1=set(list(G[i]))
    nb2=set(list(G[j]))
    nb=list(nb1&nb2)
    num=0
    for k in nb:
        if is_motif(i,j,k):
            num+=1
    return num


'''
initialize the network and regularize the adjacency matrix
'''
def initial_network(f1,f2):
    global DG,Cluster,G
    dic=dict()
    n=len(Cluster)
    for i in range(n):
        dic[Cluster[i]]=i
    DG1=nx.DiGraph()
    nn=128
    for i in range(nn):
        names=f1.readline()
        DG1.add_node(i,name=names,color=0)
    for line in f2:
        strli=line.split()
        a=int(strli[0])
        b=int(strli[1])
        DG1.add_edge(a,b)
    H=DG1.subgraph(Cluster)
    for u in H.nodes():
        DG.add_node(dic[u],name=H.nodes[u]['name'],color=0)
        G.add_node(dic[u],name=H.nodes[u]['name'],color=0)
    for e in H.edges():
        DG.add_edge(dic[e[0]],dic[e[1]])
        G.add_edge(dic[e[0]],dic[e[1]])
    A=np.zeros((n, n))
    for i in range(n):
        for j in range(i+1,n):
            num=count_motif(i,j)
            A[i,j]=num
            A[j,i]=num
    return A


'''
regularize the motif adjacency matrix
'''
def regularize_matrix(A,tau):
    global Cluster
    n=len(Cluster)
    T=np.ones((n,n))
    A=A+(tau/n)*T
    return A


'''
construct multihop matrix
'''
def multihop_matrix(A,k):
    global G
    print("hop number: "+str(k))
    N=(A.shape)[0]
    Ctmp=A
    S=np.zeros((N,N))
    S=S+A
    for i in range(k-1):
        Ctmp=np.matmul(Ctmp,A)
        S=S+Ctmp
    for i in range(N):
        for j in range(N):
            if A[i,j]!=0:
                A[i,j]=S[i,j]      
    return A


'''
multihop spectral clustering
'''
def spectral_community_detect(A,groups):
    global G
    model = SpectralClustering(n_clusters=groups,eigen_solver='arpack',random_state=56,affinity='precomputed')
    model.fit(A)
    labels=model.labels_
    return labels

        

'''
evaluate the community
'''
def evaluate_community(y_true,y_pred):
    res1=purity(y_true,y_pred)
    res2=normalized_mutual_info_score(y_true,y_pred,average_method='arithmetic')
    res3=adjusted_rand_score(y_true,y_pred)
    res4=F_score(y_true,y_pred,1)
    return res1,res2,res3,res4
    




def main():
    global G,DG
    F1=open('bay-nodes.txt', 'r')
    F2=open('bay-edges.txt','r')
    Class1=[1,1,1,2,2,2,2,2,2,3,4,4,5,5,5,5,5,6,6,5,5,3,3,3,7,8,7,7,3,9,10,10,10,11,11,12,12,12,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,7,5,4,7,6,7,7,6]
    Class2=[1,1,1,1,1,1,1,1,1,2,3,3,4,4,4,4,4,5,5,4,4,2,2,2,6,5,6,6,2,7,7,7,7,7,7,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,6,4,3,6,5,6,6,5]
    A=initial_network(F1,F2)
    tau=2.5
    A=regularize_matrix(A,tau)
    q=2
    k=4
    A=multihop_matrix(A,q)
    pred=spectral_community_detect(A,k)
    purity,nmi,ari,f1=evaluate_community(Class1,pred)
    print("class1 Purity: "+str(purity))
    print("class1 NMI: "+str(nmi))
    print("class1 ARI: "+str(ari))
    print("class1 F1: "+str(f1))
    print("#########################")
    purity,nmi,ari,f1=evaluate_community(Class2,pred)
    print("class2 Purity: "+str(purity))
    print("class2 NMI: "+str(nmi))
    print("class2 ARI: "+str(ari))
    print("class2 F1: "+str(f1))        
    

   


main()
