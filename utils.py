import os
import json
from nltk.metrics.distance import jaro_similarity
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections.abc import Iterable

# instantiate cmudict path
global cmudict_path
cmudict_path =  os.path.join('cmudict', 'cmudict.dict')

######################################################## UTILITY
class WordDict:
    def __init__(self):
        """Initialize class and create data directory

        """
        self.cmudict_path = cmudict_path
        self.data_path = self._make_data_dir()

    def _make_data_dir(self) -> str:
        """Construct the directory to house cached words

        Returns:
            str: path to the cached directory
        """
        dir = "./__cache__"
        if not os.path.isdir(dir):
            os.makedirs(dir)
        return dir

    def _create(self) -> dict:
        """Create the words dictionary based on word-syllables key-value pairs

        Returns:
            dict: dictionary consisting on word-syllables key-value pairs
        """
        words = {}
        with open(self.cmudict_path, 'r') as cmu:
            for line in cmu.readlines():
                split_line = line.rstrip('\n').split(" ")
                split_word, syllables = split_line[0], split_line[1:]
                words[split_word] = syllables
        return words

    @property
    def stream(self) -> dict:
        """Load if created, create if non-existent.

        Returns:
            dict: dictionary consisting on word-syllables key-value pairs
        """
        path = os.path.join(self.data_path, 'wordframe.json')
        if os.path.exists(path):
            with open(path, 'r') as fp:
                data = json.load(fp)
            return data

        # data doesn't exist, save out to folder
        print("Saving to cache...")
        word_dict = self._create()
        with open(path, 'w') as fp:
            json.dump(word_dict, fp)
        
        return word_dict

class Network(WordDict):
    def __init__(self):
        """Establish base network class and designations based on user
        """
        super().__init__()

    def _find_similarity(self, syls1: list[str], syls2: list[str]) -> float:
        """Find similarity using Jaro Similarity

        Args:
            syls1 (list[str]): list of syllables
            syls2 (list[str]): list of syllables

        Returns:
            float: similarity score
        """
        return jaro_similarity(syls1, syls2)
    
    def _produce_edgelist(self, n1: str, n2: str, weight: float, w1_syllables: list, w2_syllables: list) -> str:
        """Produce connection into edgelist format

        Args:
            n1 (str): node 1
            n2 (str): node 2
            weight (float): similarity as edge weight
            w1_syllables (list): list of n1 syllables
            w2_syllables (list): list of n2 syllables

        Returns:
            str: edgelist string format
        """

        w1_syl_str = ','.join(w1_syllables)
        w2_syl_str = ','.join(w2_syllables)
        return f"{n1} {n2} {weight} {w1_syl_str} {w2_syl_str}"

    def _calculate_pairwise_sim(self, combos: Iterable[tuple[str, str]]) -> dict:
        """Calculate the similarity between arbitrary words using their syllables

        Args:
            combos (Iterable[tuple[str, str]]): Combinations given the class designation

        Returns:
            dict: graph dictionary
        """
        graph_dict = {}
        for combo in tqdm(combos, total = len(combos)):
            w1, w2 = combo
            try:
                w1_syllables, w2_syllables = self.stream[w1], self.stream[w2]
                id = "_".join([w1, w2])
                if graph_dict.get(id) is not None:
                    continue
                sim = self._find_similarity(w1, w2)
                if sim <= 0:
                    continue
                graph_dict[id] = self._produce_edgelist(w1, w2, sim, w1_syllables, w2_syllables)
            except (ValueError, KeyError):
                continue
        return graph_dict 

    def produce(self, combos: Iterable[tuple[str, str]]) -> dict:
        """Produce the dictionary to construct edgelist

        Returns:
            dict: dictionary of id "word1_word2" (sorted) and edgelist configuration
        """
        # create data dict
        graph_dict = self._calculate_pairwise_sim(list(combos)) 
        return graph_dict

class Graph(Network):
    def __init__(self, cls, **kwargs):
        super().__init__()
        self.cls = cls(**kwargs)
        self.combos = list(self.cls._find_combinations)
        self.graph_dict = self.produce(self.combos)

    def __len__(self):
        return len(self.graph_dict)

    @property
    def graph(self):
        """Extract the graph given the criteria. It is not recommended to run the graph as is like ``graph()``. This 
        will construct all combinations of roughly ~160K words which generates around ~9 Billion pairs. If you would
        like to see how many edges at most will be constructed, use the ``find_max_edges()`` function for guidance.

        Args:
            n (strorint, optional): number of words to create permutations of. Defaults to None.
            seq (str, optional): letter(s) to determine what the filter should look for. Defaults to None.
            startswith (bool, optional): let class know to search using startswith or endswith. Defaults to True.
            show (bool, optional): let function know to plot the graph.       

        Returns:
            _type_: _description_
        """
        data = (('weight', float,),('n1_syllables', str,), ('n2_syllables', str,))
        graph = nx.parse_edgelist(list(self.graph_dict.values()), nodetype = str, data = data)
        return graph

def show_graph(G: nx.Graph, cmap:plt.cm = plt.cm.plasma) -> None:
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