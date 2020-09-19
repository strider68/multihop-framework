import random as rd
import numpy as np
import networkx as nx
from numpy import linalg as LA
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import metrics

'''
construct modularity matrix based on
multi-hop adjacency matrix A of graph G
'''
def construct_modularity_matrix(G,A,q,gamma,lmbda):
    N=(A.shape)[0]
    matrix_tmp=A
    S=np.zeros((N,N))
    S=S+A
    for i in range(q-1):
        matrix_tmp=np.matmul(matrix_tmp,A)
        S=S+matrix_tmp
    S=gamma*S
    das=np.zeros(N)
    W=0.0
    for i in range(N):
        for j in range(N):
            if A[i,j]!=0:
                A[i,j]=S[i,j]
                G[i][j]["weight"]=A[i,j]
                W+=S[i,j]
                das[i]+=S[i,j]
    B=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            B[i,j]=A[i,j]-((das[i]*das[j])/(1.0*W))*lmbda       
    return B,W
                    

'''
construct matrix Bg
'''
def construct_matrix_Bg(B,group):
    global G
    n=len(group)
    Bg=np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i!=j:
                Bg[i,j]=B[group[i],group[j]]
            else:
                sums=0
                for k in range(n):
                    sums+=B[group[i],group[k]]
                Bg[i,j]=B[group[i],group[j]]-sums
    return Bg



'''
compute modularity
'''
def compute_Q(s,BB,W):
    dim=BB.shape
    n=dim[0]
    m=dim[1]
    res=0
    for i in range(n):
        for j in range(m):
            res+=s[i]*BB[i,j]*s[j]
    res=res/(2*W*1.0)
    return res



'''
recursive find partitions
'''
def recursive_bisection(B,group,W):
    group=list(group)
    Bg=construct_matrix_Bg(B,group)
    w,v=LA.eigh(Bg)
    n=len(w)
    v1=v[:,n-1]
    s=[]
    for u in v1:
        if u>0:
            s.append(1)
        elif u<0:
            s.append(-1)
        else:
            rd.seed()
            r=rd.random()
            if r<=0.5:
                s.append(1)
            else:
                s.append(-1)
    Qd=compute_Q(s,Bg,W)
    res=[]
    if Qd>0:
        g1=[]
        g2=[]
        for i in range(n):
            if s[i]>0:
                g1.append(group[i])
            else:
                g2.append(group[i])
        if len(g1)<n and len(g1)>1:
            groups1=recursive_bisection(B,g1,W)
            for g in groups1:
                res.append(g)
        elif len(g1)>0:
            res.append(g1)
        if len(g2)<n and len(g2)>1:
            groups2=recursive_bisection(B,g2,W)
            for g in groups2:
                res.append(g)
        elif len(g2)>0:
            res.append(g2)
    else:
        res.append(group)
    return res
    
    
            
    
'''
detect community through multihop leading eigenvectors
of modularity matrix
'''
def multihop_lead_eigen(G,q=1,gamma=1,lmbda=1):
    A=nx.adjacency_matrix(G)
    A=A.todense()
    B,W=construct_modularity_matrix(G,A,q,gamma,lmbda)
    w,v=LA.eigh(B)
    n=len(w)
    v1=v[:,n-1]
    g1=[]
    g2=[]
    s=[]
    for u in v1:
        if u>0:
            s.append(1)
        elif u<0:
            s.append(-1)
        else:
            rd.seed()
            r=rd.random()
            if r<=0.5:
                s.append(1)
            else:
                s.append(-1)
    for i in range(n):
        if s[i]>0:
            g1.append(i)
        else:
            g2.append(i)
    groups=[g1,g2]
    res=[]
    if len(groups[0])<n and len(groups[0])>1:
        groups1=recursive_bisection(B,groups[0],W)
        for g in groups1:
            res.append(g)
    elif len(groups[0])>0:
        res.append(groups[0])
    if len(groups[1])<n and len(groups[1])>1:
        groups2=recursive_bisection(B,groups[1],W)
        for g in groups2:
            res.append(g)
    elif len(groups[1])>0:
        res.append(groups[1])
    return res

        

'''
initialize the network data
'''
def initial_network():
    TG=nx.karate_club_graph()
    vs=list(TG.nodes())
    N=len(vs)
    dic=dict()
    G=nx.Graph()
    for i in range(N):
        G.add_node(i,name=vs[i])
        dic[vs[i]]=i
    es=list(TG.edges())
    for e in es:
        G.add_edge(dic[e[0]],dic[e[1]],weight=1)
    return G



def main():
    q=3
    gamma=1
    lmbda=1
    G=initial_network()
    res=multihop_lead_eigen(G,q,gamma,lmbda)
    for r in res:
        print(sorted(r))
        print("##########################")
    
            
              
    


main()
