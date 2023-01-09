from dataclasses import dataclass
import networkx as nx
import matplotlib.pyplot as plt

@dataclass
class Graph:
    graph_data: dict

    def __len__(self):
        return len(self.graph_data)

    def __post_init__(self):
        data = (('weight', float,), ('n1_syllables', str,), ('n2_syllables', str,))
        self.graph = nx.parse_edgelist(list(self.graph_data.values()), nodetype = str, data = data)

    def show_graph(self, G: nx.Graph, cmap:plt.cm = plt.cm.plasma) -> None:
        """Graph plotting function to output given then constructed graph. Adapted from:
        https://networkx.org/documentation/stable/auto_examples/drawing/plot_directed.html?highlight=colorbar

        Args:
            G (nx.Graph): graph constructed from dictionary

        Returns:
            None
        """
        seed = 1885
        pos = nx.spring_layout(G, k = 0.15, iterations = 20, seed = seed)
        edge_colors = [G[u][v]['weight'] for u, v in G.edges]
        cmap = plt.cm.plasma

        _ = nx.draw_networkx_nodes(G, pos, node_color="pink")
        edges = nx.draw_networkx_edges(
            G,
            pos,
            edge_color=edge_colors,
            edge_cmap=cmap,
            width=2,
        )
        _ = nx.draw_networkx_labels(
            G, 
            pos
        )
        edges.set_array(edge_colors)
        cbar = plt.colorbar(edges)
        cbar.ax.set_title("Jaro Similarity")

        ax = plt.gca()
        ax.set_axis_off()
        plt.show()
        return None