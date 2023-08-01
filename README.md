# Speaker verification
This repo contains experiments in speaker verification topic. The research analyze the impact of lexical contents in phrase, pass-phrase or text in speaker verification system. Therefore, our research focuses on the experiments. For each pass-phrase, we train and evaluate a model.

## Install
Install the necessary packages for the research.
```
conda create -n speaker-verification python==3.8
conda activate speaker-verification
pip install -r requirements.txt
```
## Running

### Notebooks
- Prepare AudioMNIST data: split data into development and evaluation set. *notebooks/prepare_audio_mnist_data.ipynb*
- Generate data for experiment: *notebooks/generate_data_for_experiments.ipynb*
- Generate verifiation files for evaluation: *notebooks/generate_verification_files.ipynb*
- Parse result from checkpoints: *notebooks/parse_result.ipynb*
- Visualize the results: *notebooks/visualuzation.ipynb*

### Training and Evaluating

#### Training
Prepare dataset for training:
- VoxCeleb2: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html
- MUSAN: https://www.openslr.org/17/
- RIRs: https://www.openslr.org/28/
- AudioMNIST: https://github.com/soerenab/AudioMNIST

You can adjust your dataset path and configuration at *configs/configs.yaml*
```
python train_model.py -dataset_name=audio_mnist -stage=2 -df_train=new_data/loop1/0/development.csv -info_data=meta_data/splitted.json
```
#### Evaluation

You can adjust about something such as path to checkpoint, outputfile, and verification file at *run_eval.py* :D.

```
python run_eval.py
```
## Checkpoints

```
.
└── checkpoint_folder/
    ├── non_pretrained/
    │   ├── channel_1024/
    │   │   ├── exp1/
    │   │   │   ├── loop1/
    │   │   │   │   ├── 0/
    │   │   │   │   │   ├── <evaluation_result>.json
    │   │   │   │   │   └── ...
    │   │   │   │   └── ...
    │   │   │   ├── loop2/
    │   │   │   │   ├── 0_0/
    │   │   │   │   │   ├── <evaluation_result>.json
    │   │   │   │   │   └── ...
    │   │   │   │   └── ...
    │   │   │   ├── loop3/
    │   │   │   │   ├── 0_0_0/
    │   │   │   │   │   ├── <evaluation_result>.json
    │   │   │   │   │   └── ...
    │   │   │   │   └── ...
    │   │   │   ├── loop4
    │   │   │   └── loop5
    │   │   ├── exp2
    │   │   └── exp3
    │   ├── channel_128
    │   └── channel_64
    └── pretrained/
        ├── channel_1024
        ├── channel_128
        └── channel_64
```
Please contact me by email v.hbaoduy@gmail.com to get checkpoints as 
well as the results. You can see my implementation for basic deployment through https://github.com/vhbaoduy/voice-authentication.
# Reference
Reference from github : https://github.com/TaoRuijie/ECAPA-TDNN