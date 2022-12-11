import random

import pandas as pd
import os
import json
import numpy as np
import argparse

import utils

URL_RIR = 'http://www.openslr.org/resources/28/rirs_noises.zip'

GG_SPEECH_V2 = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
GG_SPEECH_V1 = 'http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz'


def prepare_data(root_dir, out_dir, split_ratio, dataset='arabic', df_path=None, info_path=None, verification_num=None,
                 stage=1,
                 seed=2022):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    np.random.seed(seed)
    random.seed(seed)

    infor = {}
    data = {
        'file': [],
        'word': [],
        'speaker': [],
    }
    if stage == 1:
        path = os.path.join(out_dir)
        if not os.path.exists(path):
            os.mkdir(path)

        if dataset == 'arabic':
            words = os.listdir(os.path.join(root_dir))
            for word in words:
                files = os.listdir(os.path.join(root_dir, word))
                for file in files:
                    if file.endswith('.wav'):
                        att = file[:-4].split('_')
                        data['word'].append(word)
                        data['speaker'].append(str(int(att[0])))
                        data['file'].append(word + '/' + file)

            infor['words'] = sorted(list(set(data['word'])))
            infor['speakers'] = sorted(list(set(data['speaker'])))
            infor['n_samples'] = 10

            df = pd.DataFrame(data)
            df.to_csv(os.path.join(path, 'data.csv'), index=False)
        
        elif dataset == 'audio_mnist':
            speakers = os.listdir(root_dir)
            for speaker in speakers:
                if os.path.isdir(os.path.join(root_dir, speaker)):
                    files = os.listdir(os.path.join(root_dir, speaker))
                    for file in files:
                        if file.endswith('.wav'):
                            att = file[:-4].split('_')
                            data['word'].append(att[0])
                            data['speaker'].append(str(att[1]))
                            data['file'].append(speaker + '/' + file)

            infor['words'] = sorted(list(set(data['word'])))
            infor['speakers'] = sorted(list(set(data['speaker'])))
            infor['n_samples'] = 50

            df = pd.DataFrame(data)
            df.to_csv(os.path.join(path, 'data.csv'), index=False)

        elif dataset.startswith('gg-speech'):
            df = pd.read_csv(df_path)
            info = utils.read_json(info_path)

            infor['words'] = sorted(info['words'])
            infor['speakers'] = sorted(info['speakers'])
            infor['n_samples'] = info['n_samples']

        with open(os.path.join(path, 'info.json'), 'w') as fout:
            json.dump(infor, fout)
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
        train.to_csv(os.path.join(path, 'train.csv'), index=False)
        val.to_csv(os.path.join(path, 'val.csv'), index=False)

    if stage == 2:
        path = out_dir
        if not os.path.exists(path):
            os.mkdir(path)

        infor = utils.read_json(info_path)
        idx_split = int(len(infor['speakers']) * split_ratio)
        np.random.shuffle(infor['speakers'])
        speaker_train = infor['speakers'][:idx_split]
        speaker_valid = infor['speakers'][idx_split:]
        print(len(speaker_valid), len(speaker_train))
        train = {
            'speaker': [],
            'file': []
        }
        print('Creating train data....')
        for sp in speaker_train:
            files = os.listdir(os.path.join(root_dir, str(sp)))
            for f in files:
                train['file'].append(sp + '/' + f)
                train['speaker'].append(sp)

        df_train = pd.DataFrame(train)
        df_train.to_csv(os.path.join(path, 'train.csv'), index=False)
        print('Total train: ', len(df_train))
        print('Saved at', os.path.join(path, 'train.csv'))
        print('Creating valid data...')
        assert verification_num < infor['n_samples']
        if verification_num == 0:
            verification_num = infor['n_samples'] - 1
        f = open(os.path.join(path, 'verification.txt'), 'w')

        cnt = 0
        for sp in speaker_valid:
            files = os.listdir(os.path.join(root_dir, sp))
            np.random.shuffle(files)
            for file in files:
                # print(len(list(set(files) - set([file]))), sp, file)
                pos_files = random.sample(list(set(files) - set([file])), verification_num)
                for pf in pos_files:
                    f.write("%s %s %s\n" % (str(1), sp + '/' + file, sp + '/' + pf))

                neg_files = []
                while len(neg_files) < verification_num:
                    sp_dif = random.choice(list(set(speaker_valid) - set([sp])))
                    sample = random.choice(os.listdir(os.path.join(root_dir, sp_dif)))
                    nf = sp_dif + '/' + sample
                    if nf not in neg_files:
                        neg_files.append(nf)
                for nf in neg_files:
                    f.write("%s %s %s\n" % (str(0), sp + '/' + file, nf))

                cnt += len(pos_files) + len(neg_files)
        print('Total: %s, Saved at %s' % (str(cnt), os.path.join(path, 'verification.txt')))
        f.close()

        readme = {
            'speaker_train': speaker_train,
            'speaker_valid': speaker_valid,
        }
        with open(os.path.join(path, 'info.json'), 'w') as fout:
            json.dump(readme, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prepare data file')
    parser.add_argument('-root_dir', type=str, default='./dataset', help='path to root dir dataset')
    parser.add_argument('-out_dir', type=str, default='./output', help='path to output dir')
    parser.add_argument('-split_ratio', type=float, default=0.6, help='split ratio in train and valid')
    parser.add_argument('-dataset_name', type=str, default='arabic',
                        choices=['arabic', 'gg-speech-v0.1', 'gg-speech-v0.2', 'audio_mnist'])
    parser.add_argument('-filtered_df', type=str, default='./data/gg-speech-v0.1/digits/df_filter_5.csv',
                        help='path to filtered df gg-speech-command')
    parser.add_argument('-info_path', type=str, default='./data/gg-speech_v0.1/digits/filter_5.json',
                        help='path to info.json gg-speech-command')
    # parser.add_argument('-noise_path', type=str, default ='./noise_data')
    parser.add_argument('-stage', type=int, choices=[1, 2], default=2)
    parser.add_argument('-verification_num', type=int, default=0)
    parser.add_argument('-seed', type=int, default=2022)
    args = parser.parse_args()

    prepare_data(root_dir=args.root_dir,
                 out_dir=args.out_dir,
                 split_ratio=args.split_ratio,
                 dataset=args.dataset_name,
                 df_path=args.filtered_df,
                 info_path=args.info_path,
                 stage=args.stage,
                 verification_num=args.verification_num,
                 seed=args.seed)
