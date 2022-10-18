import pandas as pd
import os
import json
import numpy as np
import argparse

URL_RIR = 'http://www.openslr.org/resources/28/rirs_noises.zip'
GG_SPEECH_V2 = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
GG_SPEECH_V1 = 'http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz'


def prepare_data(root_dir, out_dir, split_ratio, dataset='arabic', label_type=None, df_path=None, info_path=None,
                 seed=2022):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    infor = {}
    data = {
        'file': [],
        'word': [],
        'speaker': [],
    }
    if dataset == 'arabic':
        words = os.listdir(os.path.join(root_dir))
        for word in words:
            files = os.listdir(os.path.join(root_dir, word))
            for file in files:
                if file.endswith('.wav'):
                    att = file[:-4].split('_')
                    data['word'].append(word)
                    data['speaker'].append(int(att[0]))
                    data['file'].append(word + '/' + file)

        infor['words'] = sorted(list(set(data['word'])))
        infor['speakers'] = sorted(list(set(data['speaker'])))

        df = pd.DataFrame(data)
        df.to_csv(os.path.join(out_dir, 'data.csv'), index=False)

    elif dataset.startswith('gg-speech'):
        df = pd.read_csv(df_path)
        with open(info_path, 'r') as fin:
            info = json.load(fin)

        infor['words'] = sorted(info['words'])
        infor['speakers'] = sorted(info['speakers'])

    with open(os.path.join(out_dir, 'info.json'), 'w') as fout:
        json.dump(infor, fout)
    np.random.seed(seed)
    train = pd.DataFrame()
    val = pd.DataFrame()
    for word in infor['words']:
        for speaker in infor['speakers']:
            temp = df[(df['word'] == word) & (df['speaker'] == speaker)]
            total = len(temp)
            # print(temp.values)
            np.random.shuffle(temp.values)
            idx_split = int(total * split_ratio)
            train = pd.concat([train, temp[:idx_split]], ignore_index=True)
            val = pd.concat([val, temp[idx_split:]], ignore_index=True)

    print("Train: %d, Valid: %d" % (len(train), len(val)))
    print(train.dtypes)
    train.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    val.to_csv(os.path.join(out_dir, 'val.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prepare data file')
    parser.add_argument('-root_dir', type=str, default='./dataset', help='path to root dir dataset')
    parser.add_argument('-out_dir', type=str, default='./output', help='path to output dir')
    parser.add_argument('-split_ratio', type=float, default=0.6, help='split ratio in train and valid')
    parser.add_argument('-dataset_name', type=str, default='arabic',
                        choices=['arabic', 'gg-speech-v0.01', 'gg-speech-v0.02'])
    parser.add_argument('-label_type', type=str, choices=['digit', 'iot'])
    parser.add_argument('-filtered_df', type=str, help='path to filtered df gg-speech-command')
    parser.add_argument('-info_path', type=str, help='path to info.json gg-speech-command')
    # parser.add_argument('-noise_path', type=str, default ='./noise_data')
    parser.add_argument('-seed', type=int, default=2022)
    args = parser.parse_args()

    prepare_data(root_dir=args.root_dir,
                 out_dir=args.out_dir,
                 split_ratio=args.split_ratio,
                 dataset=args.dataset_name,
                 label_type=args.label_type,
                 df_path=args.filtered_df,
                 info_path=args.info_path,
                 seed=args.seed)
