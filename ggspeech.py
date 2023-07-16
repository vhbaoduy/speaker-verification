import json
import pandas as pd
import numpy as np
import os
import argparse



"""
    This file is used for google speech command
"""

WORDS = {
    'mix': ["five", "four", "nine", "one", "seven", "six",
            "three", "two", "zero", "eight", 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'],
    'digit': ["five", "four", "nine", "one", "seven", "six",
               "three", "two", "zero", "eight"],
    'iot': ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'],
}


# json_name = 'filter_5.json'
# folders = ['./gg-speech-v0.1', './gg-speech-v0.2']
def create_df(root_dir):
    folders = os.listdir(root_dir)
    data = {
        'word': [],
        'speaker': [],
        'file': [],
    }
    for fol in folders:
        path = os.path.join(root_dir, fol)
        if os.path.isdir(path) and fol != '_background_noise_':
            files = os.listdir(path)
            for file in files:
                if file.endswith('.wav'):
                    parsing = file.split("_")
                    speaker = parsing[0]
                    file_name = fol + "/" + file
                    data["file"].append(file_name)
                    data["speaker"].append(speaker)
                    data["word"].append(fol)

    return pd.DataFrame(data)


def filter_data(df, words, n_samples):
    for i, word in enumerate(words):
        temp = df[df['word'] == word]
        temp = temp.groupby('speaker').count()
        temp = temp[temp['word'] >= n_samples]
        if i == 0:
            speakers = set(temp.index.to_list())
        else:
            speakers = speakers & set(temp.index.to_list())
    print('The number of speakers: %d' % len(speakers))
    info = {
        'speakers': list(speakers),
        'words': words,
        'n_samples': n_samples,
    }
    return info


def process_data(root_dir, out_dir, words, n_samples, seed=2022):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    np.random.seed(seed)

    df = create_df(root_dir)
    df.to_csv(os.path.join(out_dir, 'data.csv'), index=False)

    info = filter_data(df, words, n_samples)
    with open(os.path.join(out_dir, 'info_filter_%s.json' % n_samples), 'w') as out:
        json.dump(info, out)

    n = info['n_samples']
    result = pd.DataFrame()
    for label in info['words']:
        for speaker in info['speakers']:
            temp = df[(df['word'] == label) & (df['speaker'] == speaker)]
            np.random.shuffle(temp.values)
            result = pd.concat([result, temp[:n]], ignore_index=True)
    print('The number of samples %d' % len(result))

    result.to_csv(os.path.join(out_dir, 'data_filter_%s.csv' %
                  n_samples), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Prepare and filter google speech command dataset')
    parser.add_argument('-root_dir', type=str,
                        help='path to dataset ./speech_commands_v0.01')
    parser.add_argument('-out_dir', type=str, help='path to folders')
    parser.add_argument('-n_samples', type=int,
                        help='the number of samples per speakers for each words', default=5)
    parser.add_argument('-type', type=str,
                        choices=['mix', 'digit', 'iot'], default='mix')
    parser.add_argument('-seed', type=int, default=2022)
    args = parser.parse_args()

    process_data(args.root_dir,
                 args.out_dir,
                 WORDS[args.type],
                 args.n_samples, seed=args.seed)
