from .scripts.graphs import Random, WordList, Synonym

def graph_random(n:int = 10, seq:str = None, startswith:bool = None, show:bool = False):
    graph = Random(n = n, seq = seq, startswith = startswith).find
    if show: 
        graph.show_graph()
    return graph

def graph_wordlist(wordlist: list, communities: bool = None, show:bool = False):
    graph = WordList(wordlist = wordlist, communities=communities).find
    if show: 
        graph.show_graph()
    return graph

def graph_synonym(word: str, communities: bool = None, n_synonyms: int = None, n:int = 10, show:bool = False):
    graph = Synonym(word = word, communities = communities, n_synonyms = n_synonyms, n = n).find
    if show: 
        graph.show_graph()
    return graph