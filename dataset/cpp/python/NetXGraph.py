import networkx as nx


def CreateNetGraph(h):
    G = nx.MultiDiGraph()
    topics="abcdefghijklmnop"
    for t in range(h.shape[0]):
        G.add_node(topics[t], bipartite=0)
    for tweet in range(h.shape[1]):
        max = 0
        maxInd = 0
        for t in range(h.shape[0]):
            if h[t, tweet] > max:
                max = h[t, tweet]
                maxInd = t
        G.add_node(tweet, bipartite=1, color = maxInd)

    count = 0
    for t in range(h.shape[0]):
        for tweet in range(h.shape[1]):
            if h[t, tweet] > 0:
                G.add_edge(topics[t], tweet, weight=h[t, tweet])
                count +=1
    # print(count)

    nx.write_gml(G, "graph.gml")
    return G

