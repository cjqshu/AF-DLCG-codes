import networkx as nx
import pandas as pd
import numpy as np
import graphviz

from utils.graph2adj import causal_graph_to_adjacency


def adjacency_to_causal_graph(df_adj):

    G = nx.DiGraph()
    nodes = list(df_adj.columns)
    G.add_nodes_from(nodes)

    for i, parent in enumerate(nodes):
        for j, child in enumerate(nodes):
            if df_adj.iloc[i, j] == 1:
                G.add_edge(parent, child)

    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))

    df_graph = G

    pos = nx.circular_layout(df_graph)

    nx.draw(df_graph, pos, with_labels=True, node_color='lightblue', node_size=2000,
            font_size=12, arrowsize=20, font_weight='bold')

    plt.title("Causal DAG from Adjacency Matrix")
    plt.show()

    A = nx.nx_agraph.to_agraph(G)
    dot_str = A.to_string()
    dot = graphviz.Source(dot_str, engine="circo")
    
    return G, dot

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
    G, dot = adjacency_to_causal_graph(df_adj)

    print(f"{df_adj} \n\n {G}")

    
