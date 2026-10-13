import seaborn as sns
import matplotlib.pyplot as plt


def plot_causal_heatmap(df, title="Causal Adjacency Matrix Heatmap", xlabel="Effect ($j$)", ylabel="Cause ($i$)"):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['text.usetex'] = True

    sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True, linewidths=0.5, square=True)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()
    plt.show()

    return


if __name__ == "__main__":

    import pandas as pd
    import numpy as np

    df = pd.DataFrame(np.array([[0, 1.1, 0], [2.2, 0, 1], [0, 3.3, 0]]), index=["A", "B", "C"], columns=["A", "B", "C"])
    plot_causal_heatmap(df, title="Causal Adjacency Matrix Heatmap", xlabel="Effect ($j$)", ylabel="Cause ($i$)")