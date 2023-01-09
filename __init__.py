import json
import os

def make_data_dir(cache_dir: str = "./__cache__") -> str:
    """Construct the directory to house cached words

    Returns:
        str: path to the cached directory
    """
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir

def establish_word_dict(cmudict_path:str = os.path.join('cmudict', 'cmudict.dict'), cache_dir: str = "./__cache__"):

    path = os.path.join(cache_dir, 'wordframe.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        return data

    # data doesn't exist, save out to folder
    print("Saving to cache...")
    word_dict = {}
    with open(cmudict_path, 'r', encoding = 'utf-8') as cmu:
        for line in cmu.readlines():
            split_line = line.rstrip('\n').split(" ")
            split_word, syllables = split_line[0], split_line[1:]
            word_dict[split_word] = syllables

    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(word_dict, fp)
    return None

def main():
    make_data_dir()
    establish_word_dict()

if __name__ == "__main__":
    main()