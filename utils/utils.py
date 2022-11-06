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
    src = 'F:\\Datasets\\speech_commands_v0.01/five/0b77ee66_nohash_0.wav'
    # des = './temp3.wav'
    # wave_files = ['five/00b01445_nohash_1.wav', 'five/00b01445_nohash_1.wav']
    # combine_waves(src,
    #               des,
    #               wave_files)
    audio, sr = load_audio(src, 16000)
    print(len(audio), sr)
