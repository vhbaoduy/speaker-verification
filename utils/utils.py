import librosa
import yaml


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
