import os
import json
from nltk.metrics.distance import jaro_similarity
from itertools import combinations
from tqdm import tqdm
import random
import networkx as nx
import matplotlib.pyplot as plt
from collections.abc import Iterable

cmudict_path =  os.path.join('cmudict', 'cmudict.dict')
class WordDict:
    def __init__(self, cmudict_path: str = cmudict_path):
        self.cmudict_path = cmudict_path
        self.data_path = self._make_data_dir()

    def _make_data_dir(self) -> str:
        dir = "./__cache__"
        if not os.path.isdir(dir):
            os.makedirs(dir)
        return dir

    def _create(self) -> dict:
        words = {}
        with open(self.cmudict_path, 'r') as cmu:
            for line in cmu.readlines():
                split_line = line.rstrip('\n').split(" ")
                split_word, syllables = split_line[0], split_line[1:]
                words[split_word] = syllables
        return words

    @property
    def stream(self) -> dict:
        path = os.path.join(self.data_path, 'wordframe.json')
        if os.path.exists(path):
            with open(path, 'r') as fp:
                data = json.load(fp)
            return data

        # data doesn't exist, save out to folder
        print("Saving to cache...")
        self.word_dict = self._create()
        with open(path, 'w') as fp:
            json.dump(self.word_dict, fp)
        
        return self.word_dict

class Network(WordDict):
    def __init__(self, cmudict_path: str = cmudict_path, n: str or int = None, seq: str = None, startswith: bool = True, **kwargs):
        super().__init__(cmudict_path)
        self.n = n
        self.seq = seq
        self.startswith = startswith

    def _find_similarity(self, syls1: list[str], syls2: list[str]) -> float:
        return jaro_similarity(syls1, syls2)

    def _find_startswith(self, keys: list[str]) -> Iterable[str]:
        return sorted(filter(lambda x: x.startswith(self.seq), keys))

    def _find_endswith(self, keys: list[str]) -> Iterable[str]:
        return sorted(filter(lambda x: x.endswith(self.seq), keys))

    @property
    def _find_combinations(self) -> Iterable[tuple[str, str]]:
        stream_keys = list(self.stream.keys())
        if not self.n and not self.seq:
            final_keys = stream_keys
        elif self.n and not self.seq:
            final_keys = random.sample(stream_keys, self.n)
        elif not self.n and self.seq:
            if self.startswith:
                final_keys = self._find_startswith(stream_keys)
            else:
                final_keys = self._find_endswith(stream_keys)
        else:
            if self.startswith:
                final_keys = random.sample(self._find_startswith(stream_keys), self.n)
            else:
                final_keys = random.sample(self._find_endswith(stream_keys), self.n)

        return combinations(final_keys, 2)

    def _calculate_pairwise_sim(self, combos) -> dict:
        graph_dict = {}
        for combo in tqdm(combos, total=self._length):
            w1, w2 = combo
            id = "_".join(list(sorted([w1, w2])))
            if graph_dict.get(id) is not None:
                continue
            sim = self._find_similarity(w1, w2)
            if sim <= 0:
                continue
            connection = {"node1": w1, 'node2': w2, 'weight': sim}
            graph_dict[id] = connection
        return graph_dict

    @property
    def _length(self) -> int:
        if self.n == 'all':
            num_keys = len(list(self.stream.keys()))
        else:
            num_keys = self.n
        return int((num_keys * (num_keys - 1)) / 2)

    @property
    def produce(self) -> dict:
        # create data dict
        graph_dict = self._calculate_pairwise_sim(self._find_combinations) 
        return graph_dict

################################### POLICY FUNCTIONS ####################################################
def _extract(**kwargs):
    network = Network(**kwargs)
    return network.produce

def _produce_edgelist(collection: dict) -> list[str]:
    edgelist = []
    for _, v in collection.items():
        n1, n2, weight = list(v.values())
        edgestring = f"{n1} {n2} " + "{" + f"\'weight\': {weight}" + "}"
        edgelist.append(edgestring)
    return edgelist

def _show_graph(G: nx.Graph) -> None:
    # https://networkx.org/documentation/stable/auto_examples/drawing/plot_directed.html?highlight=colorbar
    pos = nx.spring_layout(G, k = 0.15, iterations = 20)
    M = G.number_of_edges()
    edge_colors = [G[u][v]['weight'] for u, v in G.edges]
    node_labels = [G[u] for u in G.nodes]
    cmap = plt.cm.plasma

    _ = nx.draw_networkx_nodes(G, pos, node_color="pink", label = node_labels)
    edges = nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        edge_cmap=cmap,
        width=2,
    )
    labels = nx.draw_networkx_labels(
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

def graph(**kwargs):
    graph = nx.parse_edgelist(_produce_edgelist(_extract(**kwargs)))
    if kwargs.get('show'):
        _show_graph(graph)
    return graph