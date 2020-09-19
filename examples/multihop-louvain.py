import random as rd
import numpy as np
import networkx as nx
from Louvain import PyLouvain
from scipy.sparse import csr_matrix
from time import *

G= nx.Graph()




'''
initialize the network
'''
def initial_network():
    global G
    G.clear()
    print("facebook")
    TG=nx.read_gml('facebook.gml',label='id')
    vs=list(TG.nodes())
    N=len(vs)
    dic=dict()
    for i in range(N):
        G.add_node(i,name=vs[i])
        dic[vs[i]]=i
    es=list(TG.edges())
    for e in es:
        G.add_edge(dic[e[0]],dic[e[1]],weight=1)
    A = nx.adjacency_matrix(G)
    return A


'''
construct multihop matrix
'''
def multihop_matrix(A,k):
    global G
    print("hop number: "+str(k))
    N=G.number_of_nodes()
    beta=0.5
    Ctmp=A
    #S=csr_matrix((N,N),dtype=np.int64)
    S=beta*A
    es=list(G.edges())
    for i in range(k-1):
        Ctmp=Ctmp*A
        for e in es:
            S[e[0],e[1]]=S[e[0],e[1]]+beta*Ctmp[e[0],e[1]]
    for e in es:
        G[e[0]][e[1]]["weight"]=S[e[0],e[1]]          



'''
apply louvain method
'''
def louvian_detect():
    global G
    nodes=list(G.nodes())
    es=list(G.edges())
    edges=[]
    for e in es:
        edges.append(((e[0],e[1]),int(G[e[0]][e[1]]["weight"])))
    model=PyLouvain(nodes,edges)
    res=model.apply_method()
    groups=res[0]
    return len(groups)



        

def main():
    global G
    begin_time = time()
    hop_number=2
    A=initial_network()
    multihop_matrix(A,hop_number)
    res=louvian_detect()
    print("the number of communities: "+str(res))
    end_time = time()
    run_time = end_time-begin_time
    print("the total running time: "+str(run_time))
        
   
    
            
            
            
   
   
   
    
   
    


main()

