import pandas as pd
import networkx as nx
import nx_cugraph as nxcg


url = "https://data.rapids.ai/cugraph/datasets/cit-Patents.csv"
df = pd.read_csv(url, sep=" ", names=["src", "dst"], dtype="int32")
G = nx.from_pandas_edgelist(df, source="src", target="dst")
# G = nx.Graph()

# populate the graph
#  ...

nxcg_G = nxcg.from_networkx(G)             # conversion happens once here
nx.betweenness_centrality(nxcg_G, k=1000,backend="cugraph")  # nxcg Graph type causes cugraph backend
                                           # to be used, no conversion necessary