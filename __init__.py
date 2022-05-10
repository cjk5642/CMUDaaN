import itertools
import os
import json
from nltk.metrics.distance import jaro_similarity
from nltk.corpus import wordnet as wn
from itertools import combinations, chain
from tqdm import tqdm
import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
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
        self.word_dict = self._create()
        with open(path, 'w') as fp:
            json.dump(self.word_dict, fp)
        
        return self.word_dict

class Network(WordDict):
    def __init__(self, 
                word: str = None, 
                wordlist: list = None, 
                communities: bool = None,
                n: int = 10, 
                seq: str = None, 
                startswith: bool = True,
                n_synonyms: int = None, 
                **kwargs):
        """Establish base network class and designations based on user.

        Args:
            n (str or int, optional): number of words to create permutations of. Defaults to None.
            seq (str, optional): letter(s) to determine what the filter should look for. Defaults to None.
            startswith (bool, optional): let class know to search using startswith or endswith. Defaults to True.
        """
        super().__init__()
        self.word = word
        self.wordlist = wordlist
        self.communities = communities
        self.n = int(n)
        self.seq = seq
        self.startswith = startswith
        self.n_synonyms = n_synonyms

    def _find_similarity(self, syls1: list[str], syls2: list[str]) -> float:
        """Find similarity using Jaro Similarity

        Args:
            syls1 (list[str]): list of syllables
            syls2 (list[str]): list of syllables

        Returns:
            float: similarity score
        """
        return jaro_similarity(syls1, syls2)

    def _find_startswith(self, keys: list[str]) -> Iterable[str]:
        """Find all words that start with self.seq

        Args:
            keys (list[str]): Words to be filtered

        Returns:
            Iterable[str]: new set of words based on criteria
        """
        return set(filter(lambda x: x.startswith(self.seq), keys))

    def _find_endswith(self, keys: list[str]) -> Iterable[str]:
        """Final all words that end with self.seq

        Args:
            keys (list[str]): Words to be filtered

        Returns:
            Iterable[str]: new set of wrods based on critera
        """
        return set(filter(lambda x: x.endswith(self.seq), keys))

    def _find_combinations_driver(self, stream_keys: list):
        # if n and seq are None
        if not self.n and not self.seq:
            final_keys = stream_keys
        # if n is not None and seq is None
        elif self.n and not self.seq:
            final_keys = random.sample(stream_keys, self.n)
        # if n is None and seq is not None
        elif not self.n and self.seq:
            if self.startswith:
                final_keys = self._find_startswith(stream_keys)
            else:
                final_keys = self._find_endswith(stream_keys)
        # if n and seq are not None
        else:
            if self.startswith:
                final_keys = random.sample(self._find_startswith(stream_keys), self.n)
            else:
                final_keys = random.sample(self._find_endswith(stream_keys), self.n)
        return final_keys

    def _ensure_word_in_sample(self, word: str, stream_keys: list) -> list:
        return combinations([word] + random.sample(stream_keys, self.n), 2)

    @property
    def _find_combinations(self) -> Iterable[tuple[str, str]]:
        """Find combinations based on class criteria

        Returns:
            Iterable[tuple[str, str]]: generated combinations of words
        """
        stream_keys = list(self.stream.keys())

        # if wordlist is not None and commuinities is None
        if self.wordlist and not self.communities:
            return combinations(self.wordlist, 2)

        # if wordlist and communities is not None
        elif self.wordlist and self.communities:
            wordlist_comb = list(combinations(self.wordlist, 2)) # n*(n-1)/2
            new_comb = list(chain.from_iterable([self._ensure_word_in_sample(w, stream_keys) for w in self.wordlist])) # 
            return set(wordlist_comb + new_comb)

        final_keys = self._find_combinations_driver(stream_keys)
        return combinations(final_keys, 2)

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
            w1_syllables, w2_syllables = self.stream[w1], self.stream[w2]
            id = "_".join(list(sorted([w1, w2])))
            if graph_dict.get(id) is not None:
                continue
            sim = self._find_similarity(w1, w2)
            if sim <= 0:
                continue
            graph_dict[id] = self._produce_edgelist(w1, w2, sim, w1_syllables, w2_syllables)
        return graph_dict 
        
    @property
    def produce(self) -> dict:
        """Produce the dictionary to construct edgelist

        Returns:
            dict: dictionary of id "word1_word2" (sorted) and edgelist configuration
        """
        # create data dict
        graph_dict = self._calculate_pairwise_sim(self._find_combinations) 
        return graph_dict

################################### POLICY FUNCTIONS ####################################################
def _extract(**kwargs) -> dict:
    """Extract the Network response

    Returns:
        dict: dictionary of id "word1_word2" (sorted) and edgelist configuration
    """
    network = Network(**kwargs)
    return network.produce

def _show_graph(G: nx.Graph) -> None:
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
    node_labels = [G[u] for u in G.nodes]
    cmap = plt.cm.plasma

    nodes = nx.draw_networkx_nodes(G, pos, node_color="pink")
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

def find_max_edges(n: int) -> str:
    """Find max number of edges given n words

    Args:
        n (int): number of words or sample size

    Returns:
        str: Maximum number of edges
    """
    network = Network(n = n)
    return f"Maximum number of edges: {network._length}"

def graph(**kwargs):
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
    graph = nx.parse_edgelist(list(_extract(**kwargs).values()), nodetype = str, data = data)
    if kwargs.get('show'):
        _show_graph(graph)
    return graph

def graph_wordlist(wordlist: list, communities: bool = None, **kwargs):
    kwargs['wordlist'] = wordlist
    kwargs['communities'] = communities
    return graph(**kwargs)

def graph_synonym(word: str, communities: bool = None, n_synonyms: bool = None, **kwargs):
    kwargs['word'] = word
    kwargs['communities'] = communities
    kwargs['n_synonyms'] = n_synonyms
    return graph(**kwargs)

def _create_community(word:str, n_synonyms:int = None, n:int = None):
    # find n synonyms for word
    # find n words surround word and word synonyms
    # run jaro similarity on all words
    syn = wn.synsets(word)
    if not n_synonyms:
        if len(syn) < n_synonyms:
            syn = random.sample(syn, n_synonyms)
    
    pass