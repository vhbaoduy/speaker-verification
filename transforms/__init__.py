from .transform_wav import *
from torchvision.transforms import Compose
from datasets import AugmentationDataset, BackgroundNoiseDataset


def build_transform(audio_config,
                    mode,
                    num_stack=1,
                    noise_path=None,
                    stage=1):
    assert stage in [1,2]
    duration = audio_config['duration']
    add_sample = audio_config['add_sample']
    transform = FixAudioLength(time=duration,
                               add_sample=add_sample,
                               num=num_stack)
    if mode == 'train':
        if noise_path is not None:
            bg_dataset = AugmentationDataset(musan_path=noise_path['musan_path'],
                                             rir_path=noise_path['rir_path'])

            augment = Augmentation(bg_dataset=bg_dataset)
        return Compose([transform, augment, ToTensor('samples', 'input',mode=mode,stage=stage)])
    else:
        if stage==1:
            bg_dataset = BackgroundNoiseDataset(paths=noise_path['noise_test_paths'],
                                                sample_rate=audio_config['sample_rate'],
                                                sample_length=audio_config['duration'],
                                                add_sample=add_sample)
            add_noise = AddNoiseForTestPhase(bg_dataset=bg_dataset)
            return Compose([transform, add_noise, ToTensor('samples', 'input', mode=mode,stage=stage)])
        else:
            return Compose([transform, ToTensor('samples', 'input', mode=mode,stage=stage)])

