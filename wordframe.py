import os
import pandas as pd
import re
from sqlalchemy import create_engine

class WordFrame:
    def __init__(self, cmudict_path: str = os.path.join('cmudict', 'cmudict.dict')):
        self.cmudict_path = cmudict_path
        self.engine = create_engine('sqlite://', echo=False)
        self.word_frame = self.create()
        self.to_db()
    
    def enhance(self, wf: pd.DataFrame):
        syllables_split = wf['syllables'].str.split(", ", expand = True)
        wf = pd.concat([wf, syllables_split], axis = 1).drop("syllables", axis = 1)
        wf = pd.melt(wf, id_vars = ['word'], var_name="syllable_num", value_name="syllable") \
            .dropna() \
            .sort_values(['word', 'syllable_num']) \
            .reset_index(drop = True)

        wf['vowel'] = wf['syllable'].apply(lambda x: 1 if re.search("^[AEIOU]", x) else 0)
        return wf

    def create(self):
        words = []
        with open(self.cmudict_path, 'r') as cmu:
            for line in cmu.readlines():
                split_line = line.rstrip('\n').split(" ")
                split_word, syllables = split_line[0], split_line[1:]
                word = {"word": split_word, "syllables": ", ".join(syllables)}
                words.append(word)

        word_frame = self.enhance(pd.DataFrame.from_dict(words))
        return word_frame

    def to_db(self):
        try:
            print("Saving to database...")
            self.word_frame.to_sql('wordframe', con=self.engine)
        except ValueError:
            return None
        return None
