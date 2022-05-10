import os
import re
import json
from nltk.metrics.distance import jaro_similarity
from itertools import combinations
from tqdm import tqdm
import multiprocessing as mp

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
    def stream(self):
        path = os.path.join(self.data_path, 'wordframe.json')
        if os.path.exists(path):
            with open(path, 'r') as fp:
                data = json.load(fp)
            return data

        # data doesn't exist, save out to folder
        print("Saving to database...")
        self.word_dict = self._create()
        with open(path, 'w') as fp:
            json.dump(self.word_dict, fp)
        
        return self.word_dict

class Network(WordDict):
    def __init__(self, cmudict_path: str = cmudict_path):
        super().__init__(cmudict_path)
        self._combos = self._find_combinations()
        self._graph_dict = dict()

    def _find_similarity(self, syls1: list, syls2: list) -> float:
        return jaro_similarity(syls1, syls2)

    def _find_combinations(self):
        return combinations(list(self.stream.keys()), 2)

    def _calculate_pairwise_sim(self, w1, w2) -> tuple[str, dict]:
        id = "_".join(list(sorted([w1, w2])))
        if self._graph_dict.get(id) is not None:
            return None
        sim = self._find_similarity(w1, w2)
        if sim <= 0:
            return None
        connection = {"node1": w1, 'node2': w2, 'weight': sim}
        self._graph_dict[id] = connection
        return None

    @property
    def _length(self) -> int:
        num_keys = len(list(self.stream.keys()))
        return int((num_keys * (num_keys - 1)) / 2)

    @property
    def create(self):
        path = os.path.join(self.data_path, "graph.json")
        if os.path.exists(path):
            with open(path, 'r') as fp:
                self._graph_dict = json.load(fp)
            return self._graph_dict

        # create data dict
        _combos = self._find_combinations()
        with mp.Pool() as pool:
            _ = pool.starmap(self._calculate_pairwise_sim, tqdm(_combos, total=self._length), chunksize= 1000)
        
        # convert to dictionary
        with open(path, 'w') as fp:
            json.dump(self._graph_dict, fp)
        return self._graph_dict
