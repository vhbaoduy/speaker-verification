import json
import wave
import librosa
import yaml
import os
import soundfile


def label2index(labels: list, label: str):
    return labels.index(label)


def index2label(labels: list, idx: int):
    return labels[idx]


def load_audio(path, sample_rate):
    samples, sample_rate = librosa.load(path=path,
                                        sr=sample_rate)

    return samples, sample_rate


def load_config_file(file_path: str):
    """
    Load config file
    :param file_path: path to config
    :return:
    """
    try:
        yaml_config_file = open(file_path)
        file = yaml.load(yaml_config_file, Loader=yaml.FullLoader)
        return file
    except FileNotFoundError:
        print("Can not open config file.")
    return None


def read_json(path_to_json):
    with open(path_to_json, 'r') as fin:
        data = json.load(fin)
    return data


def combine_waves(src,
                  des,
                  wave_files):
    data = []
    for w in wave_files:
        path = os.path.join(src, w)
        sample = wave.open(path, 'rb')
        data.append([sample.getparams(), sample.readframes(sample.getnframes())])
        sample.close()

    out = wave.open(des, 'wb')
    out.setparams(data[0][0])
    for i in range(len(data)):
        out.writeframes(data[i][1])
    out.close()


if __name__ == '__main__':
    src = './data/audio_mnist'
    des = './temp.wav'
    wave_files = ['01/1_01_0.wav', '01/1_01_0.wav']
    # combine_waves(src,
    #               des,
    #               wave_files)
    # import numpy
    # import torch
    # audio, _ = soundfile.read(src)
    # # Full utterance
    # data_1 = torch.FloatTensor(numpy.stack([audio], axis=0))
    # # audio, sr = load_audio(src, 16000)
    # print(data_1.size())
    sample, sr = load_audio('temp.wav', 16000)
    from IPython import embed
    embed()
