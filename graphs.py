from utils import WordDict
from collections.abc import Iterable
import random
import nltk
import os
from itertools import combinations, chain
import networkx as nx
import matplotlib.pyplot as plt

# load wordnet
path = os.path.abspath("")
if path not in nltk.data.path:
    nltk.data.path.append(path)
from nltk.corpus import wordnet as wn
        
# load the worddict
wd = WordDict()

class Random:
    def __init__(self, n:int = 10, seq:str = None, startswith:bool = None):
        self.n = int(n) if isinstance(n, int) else None
        self.seq = seq
        self.startswith = startswith

    def __str__(self):
        return "Random"

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

    @property
    def _find_combinations(self):
        stream_keys = list(wd.stream.keys())
        final_keys = self._find_combinations_driver(stream_keys)
        return combinations(final_keys, 2)

class WordList:
    def __init__(self, wordlist:list[str], n_per_community:int = 10, communities:bool = None):
        self.wordlist = wordlist
        self.n_per_community = int(n_per_community) if n_per_community else None
        self.communities = communities

    def __str__(self):
        return "WordList"

    def _ensure_word_in_sample(self, word: str, stream_keys: list) -> list:
        return combinations([word] + random.sample(stream_keys, self.n), 2)

    @property
    def _find_combinations(self) -> Iterable[tuple[str, str]]:
        """Find combinations based on class criteria

        Returns:
            Iterable[tuple[str, str]]: generated combinations of words
        """
        # if wordlist is not None and commuinities is None
        if self.wordlist and not self.communities:
            return combinations(self.wordlist, 2)

        # if wordlist and communities is not None
        if self.wordlist and self.communities:
            stream_keys = list(wd.stream.keys())
            wordlist_comb = list(combinations(self.wordlist, 2)) # n*(n-1)/2
            new_comb = list(chain.from_iterable([self._ensure_word_in_sample(w, stream_keys) for w in self.wordlist])) # 
            return set(wordlist_comb + new_comb)

class Synonym:
    def __init__(self, word: str, communities: bool = None, n_synonyms: int = None, n:int = 10):
        self.communities = communities
        self.word = word
        self.n_synonyms = int(n_synonyms) if isinstance(n_synonyms, int) else None
        self.n = int(n) if isinstance(n, int) else None

    def __str__(self):
        return "Synonym"

    @property
    def _synonyms(self):
        syn = wn.synsets(self.word)
        if self.n_synonyms is not None:
            if self.n_synonyms < len(syn):
                syn = random.sample(syn, self.n_synonyms)
        syn = set(map(lambda x: str(x.name().split('.')[0]), syn))
        return syn

    def _ensure_word_in_sample(self, word: str, stream_keys: list) -> list:
        return combinations([word] + random.sample(stream_keys, self.n), 2)
    
    @property
    def _find_combinations(self) -> Iterable[tuple[str, str]]:
        """Find combinations based on class criteria

        Returns:
            Iterable[tuple[str, str]]: generated combinations of words
        """
        # if wordlist is not None and commuinities is None
        if not self.communities:
            return combinations(self._synonyms, 2)

        syns = self._synonyms
        stream_keys = list(wd.stream.keys())
        wordlist_comb = list(combinations(syns, 2)) # n*(n-1)/2
        new_comb = list(chain.from_iterable([self._ensure_word_in_sample(w, stream_keys) for w in syns])) # 
        return set(wordlist_comb + new_comb)