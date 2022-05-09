import os
import re
import json
from nltk.metrics.distance import jaro_similarity
from itertools import combinations

class WordDict:
    def __init__(self, cmudict_path: str = os.path.join('cmudict', 'cmudict.dict')):
        self.cmudict_path = cmudict_path
        self.data_path = self._make_data_dir()

    def _make_data_dir(self) -> str:
        dir = "./data"
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
    def __init__(self):
        super().__init__(self)
        self.combo = self._find_combinations()

    def _find_similarity(self, syls1, syls2):
        return jaro_similarity(syls1, syls2)

    def _find_combinations(self):
        return combinations(list(self.stream.keys()), 2)
