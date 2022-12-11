import utils
import argparse
import pandas as pd
import os
from tqdm import tqdm
import random
import json


def check_duplicate(words):
    if words.count(words[0]) == len(words):
        return True
    return False


def generate_file(samples, words, n_samples):
    files = []
    names = []
    if len(words) == 1:
        for i, sample in enumerate(samples[words[0]]):
            combine_list = [sample]
            idx = sample[:-4].split('_')[-1]
            name = (words[0] + '_%s_' % idx) + '.wav'

            files.append((combine_list, name))
            names.append(name)
        return files

    if check_duplicate(words):
        for i, sample in enumerate(samples[words[0]]):
            combine_list = [sample] * len(words)
            idx = sample[:-4].split('_')[-1]
            name = (words[0] + '_%s_' % idx) * len(words) + '.wav'

            files.append((combine_list, name))
            names.append(name)

    while len(files) < n_samples:
        name = ''
        combine_list = []
        for w in words:
            e = random.choice(samples[w])
            idx = e[:-4].split('_')[-1]
            name += w + '_%s_' % idx
            combine_list.append(e)
        name += '.wav'
        if name not in names:
            files.append((combine_list, name))
            names.append(name)

    return files


def create_data(config):
    # Write info:
    if not os.path.exists(config['out_dir']):
        os.mkdir(config['out_dir'])

    readme = {
        'dataset': config['dataset'],
        'n_samples': config['n_samples'],
        'samples': config['words'],
    }
    # file = open(os.path.join(config['out_dir'], 'info.txt'), 'w')
    # file.write('Combine dataset %s\n' % config['dataset'])
    # file.write('Number of samples: %d\n' % (config['n_samples']))
    # file.write('Samples: %s\n' % config['words'])

    # Read dataframe and infor
    df = pd.read_csv(config['df_path'])
    df['speaker'] = df['speaker'].map(str)
    df['word'] = df['word'].map(str)
    info = utils.read_json(config['info_path'])
    n = info['n_samples']
    assert config['n_samples'] <= n ** len(config['words'])
    # np.random.seed(seed)
    # index_split = int(len(info['speakers']) * config['ratio_split'])
    # np.random.shuffle(info['speakers'])
    # speaker_train = info['speakers'][:index_split]
    # speaker_valid = info['speakers'][index_split:]
    # print(len(speaker_valid), len(speaker_train), len(info['speakers']))
    # print(df)
    info_combine = {}

    for sp in info['speakers']:
        data = {}
        if config['dataset'] == 'audio_mnist':
            sp_str = sp
            sp = str(int(sp))
        for w in config['words']:
            
            #     w = int(w)
            filter = df[(df['speaker'] == sp) & (df['word'] == w)]
            assert len(filter) > 0
            data[w] = filter['file'].tolist()
        if config['dataset'] == 'audio_mnist':
            info_combine[sp_str] = data
        else:
            info_combine[sp] = data

    pbar = tqdm(info['speakers'])
    for i, sp in enumerate(pbar):
        # Create dir
        path = os.path.join(config['out_dir'], str(sp))
        if not os.path.exists(path):
            os.mkdir(path)
        combine_files = generate_file(info_combine[sp], config['words'], n_samples=config['n_samples'])
        for i, (files, name) in enumerate(combine_files):
            des_path = os.path.join(path, name)
            utils.combine_waves(src=config['root_dir'],
                                des=des_path,
                                wave_files=files)

            pbar.set_postfix({
                'speaker': sp,
                'Total': '%s/%s' % (i, len(combine_files))
            })

    readme['speakers'] = info['speakers']
    with open(os.path.join(config['out_dir'], 'readme.json'), 'w') as fout:
        json.dump(readme, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Combine data ...')
    parser.add_argument('-config_file', type=str, default='combine_data.yaml')
    # parser.add_argument('-seed', type=int, default=2022)

    args = parser.parse_args()

    cfgs = utils.load_config_file(os.path.join('./configs', args.config_file))
    create_data(config=cfgs)
