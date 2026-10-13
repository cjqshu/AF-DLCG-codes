import networkx as nx
import pandas as pd
import numpy as np

def causal_graph_to_adjacency(G, node_order=None, to_csv=False, filename="causal_adjacency_matrix.csv"):
    if node_order is None:
        node_order = list(G.nodes())
    n = len(node_order)
    node_index = {node: i for i, node in enumerate(node_order)}

    adj = np.zeros((n, n), dtype=int)

    for parent, child in G.edges():
        if parent in node_index and child in node_index:
            adj[node_index[parent], node_index[child]] = 1

    df = pd.DataFrame(adj, index=node_order, columns=node_order)

    if to_csv:
        df.to_csv(filename)
        print(f"the adjacency matrix is saved {filename}")


    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))

    sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True, linewidths=0.5, square=True)

    plt.title("Causal Adjacency Matrix Heatmap", fontsize=14)
    plt.xlabel("Effect (j)")
    plt.ylabel("Cause (i)")
    plt.tight_layout()
    plt.show()

    return df


if __name__ == '__main__':

    G = nx.DiGraph()

    nodes = ["Mo", "Cr", "Ni", "W", "roughness", "temperature", "heating_rate", "NaCl", "CPT"]

    edges = [
        ("Mo", "CPT"), ("Cr", "CPT"), ("Ni", "CPT"), ("W", "CPT"),
        ("heating_rate", "roughness"), ("roughness", "CPT"),
        ("temperature", "CPT"), ("NaCl", "CPT")
    ]
    G.add_edges_from(edges)

    df_adj = causal_graph_to_adjacency(G, node_order=nodes, to_csv=False)

    print(df_adj)
