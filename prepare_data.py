import pandas as pd
import os
import json
import numpy as np
import argparse


URL_RIR = 'http://www.openslr.org/resources/28/rirs_noises.zip'


def prepare_data(root_dir, out_dir, split_ratio, dataset='arabic', seed=2022):
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

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(out_dir, 'data.csv'), index=False)

    with open(os.path.join(out_dir, 'info.json'), 'w') as fout:
        json.dump(infor, fout)

    np.random.seed(seed)
    train = pd.DataFrame()
    val = pd.DataFrame()
    for word in words:
        for speaker in infor['speakers']:
            temp = df[(df['word'] == word) & (df['speaker'] == speaker)]
            total = len(temp)
            # print(temp.values)
            np.random.shuffle(temp.values)
            idx_split = int(total * split_ratio)
            train = pd.concat([train, temp[:idx_split]], ignore_index=True)
            val = pd.concat([val, temp[idx_split:]], ignore_index=True)

    print("Train: %d, Valid: %d" %(len(train), len(val)))
    print(train.dtypes)
    train.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    val.to_csv(os.path.join(out_dir, 'val.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prepare data file')
    parser.add_argument('-root_dir', type=str, default='./dataset', help='path to root dir dataset')
    parser.add_argument('-out_dir', type=str, default='./output', help='path to output dir')
    parser.add_argument('-split_ratio', type=float, default=0.6, help='split ratio in train and valid')
    parser.add_argument('-dataset_name', type=str, default='arabic')
    # parser.add_argument('-noise_path', type=str, default ='./noise_data')
    parser.add_argument('-seed', type=int, default=2022)
    args = parser.parse_args()

    prepare_data(root_dir=args.root_dir,
                 out_dir=args.out_dir,
                 split_ratio=args.split_ratio,
                 dataset=args.dataset_name,
                 # noise_path=args.noise_path,
                 seed=args.seed)

