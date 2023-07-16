# Speaker verification
This repo contains experiments in speaker verification topic. The research analyze the impact of lexical contents in phrase, pass-phrase or text in speaker verification system.

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

# Reference
Reference from github : https://github.com/TaoRuijie/ECAPA-TDNN