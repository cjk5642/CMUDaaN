from utils import Graph, show_graph
from graphs import WordList, Random, Synonym

################################### POLICY FUNCTIONS ####################################################
def graph_random(n:int = 10, seq:str = None, startswith:bool = None, show:bool = False):
    graph = Graph(cls = Random, n = n, seq = seq, startswith = startswith)
    if show: show_graph(graph.graph)
    return graph

def graph_wordlist(wordlist: list, communities: bool = None, show:bool = False):
    graph = Graph(cls = WordList, wordlist = wordlist, communities=communities)
    if show: show_graph(graph.graph)
    return graph

def graph_synonym(word: str, communities: bool = None, n_synonyms: int = None, n:int = 10, show:bool = False):
    graph = Graph(cls = Synonym, word = word, communities = communities, n_synonyms = n_synonyms, n = n)
    if show: show_graph(graph.graph)
    return graph