import os
import json
from typing import Iterable
from itertools import chain, combinations
import random

from datamodels import Graph

from tqdm import tqdm
from nltk.metrics.distance import jaro_similarity
from nltk.corpus import wordnet as wn 

def load_word_dict():
    path = os.path.join("__cache__", "wordframe.json")
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data

WORD_DICT = load_word_dict()

class Network:
    def __init__(self, combos: Iterable[tuple[str, str]]):
        self.combos = combos

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

    @property
    def produce(self) -> dict:
        """Calculate the similarity between arbitrary words using their syllables

        Args:
            combos (Iterable[tuple[str, str]]): Combinations given the class designation

        Returns:
            dict: graph dictionary
        """
        graph_dict = {}
        for (w1,w2) in tqdm(self.combos, total = len(self.combos)):
            try:
                w1_syllables, w2_syllables = WORD_DICT[w1], WORD_DICT[w2]
                wordpair_id = "_".join([w1, w2])
                if graph_dict.get(id) is not None:
                    continue
                sim = self._find_similarity(w1, w2)
                if sim <= 0:
                    continue
                graph_dict[wordpair_id] = self._produce_edgelist(w1, w2, sim, w1_syllables, w2_syllables)
            except (ValueError, KeyError):
                continue
        return Graph(graph_data = graph_dict)

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
    def find(self):
        stream_keys = list(WORD_DICT.keys())
        final_keys = self._find_combinations_driver(stream_keys)
        combos = combinations(final_keys, 2)
        return Network(combos).produce

class WordList:
    def __init__(self, wordlist:list[str], n_per_community:int = 10, communities:bool = None):
        self.wordlist = wordlist
        self.n_per_community = int(n_per_community) if n_per_community else None
        self.communities = communities

    def __str__(self):
        return "WordList"

    def _ensure_word_in_sample(self, word: str, stream_keys: list) -> list:
        return combinations([word] + random.sample(stream_keys, self.n_per_community), 2)

    @property
    def find(self) -> Iterable[tuple[str, str]]:
        """Find combinations based on class criteria

        Returns:
            Iterable[tuple[str, str]]: generated combinations of words
        """
        # if wordlist is not None and commuinities is None
        if self.wordlist and not self.communities:
            combos = combinations(self.wordlist, 2)
            return Network(combos=combos).produce

        # if wordlist and communities is not None
        if self.wordlist and self.communities:
            stream_keys = list(WORD_DICT.keys())
            wordlist_comb = list(combinations(self.wordlist, 2)) # n*(n-1)/2
            new_comb = list(chain.from_iterable([self._ensure_word_in_sample(w, stream_keys) for w in self.wordlist])) # 
            combos = set(wordlist_comb + new_comb)
            return Network(combos=combos).produce

class Synonym:
    def __init__(self, word: str, communities: bool = None, n_synonyms: int = None, n:int = 10):
        self.communities = communities
        self.word = word
        self.n_synonyms = int(n_synonyms) if isinstance(n_synonyms, int) else None
        self.n = int(n) if isinstance(n, int) else None

    def __str__(self):
        return "Synonym"

    def _ensure_word_in_sample(self, word: str, stream_keys: list) -> list:
        return combinations([word] + random.sample(stream_keys, self.n), 2)

    @property
    def _synonyms(self):
        syn = wn.synsets(self.word)
        if self.n_synonyms is not None:
            if self.n_synonyms < len(syn):
                syn = random.sample(syn, self.n_synonyms)
        syn = set(map(lambda x: str(x.name().split('.')[0]), syn))
        return syn

    @property
    def find(self) -> Iterable[tuple[str, str]]:
        """Find combinations based on class criteria

        Returns:
            Iterable[tuple[str, str]]: generated combinations of words
        """
        # if wordlist is not None and commuinities is None
        if not self.communities:
            return combinations(self._synonyms, 2)

        stream_keys = list(WORD_DICT.keys())
        wordlist_comb = list(combinations(self._synonyms, 2)) # n*(n-1)/2
        new_comb = list(chain.from_iterable([self._ensure_word_in_sample(w, stream_keys) for w in self._synonyms])) # 
        combos = set(wordlist_comb + new_comb)
        return Network(combos=combos).produce