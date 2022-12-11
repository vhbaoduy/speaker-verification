import pandas as pd
import os
import soundfile
if __name__ == '__main__':
    df = pd.read_csv('process_data/audio_mnist/0.5/data.csv')
    path = 'data/audio_mnist'
    sample ={}
    for i in range(len(df)):
        word = df.iloc[i]['word']
        if word not in sample:
            sample[word] = []
        sp = df.iloc[i]['speaker']
        if int(sp) < 10:
            sp = '0' + str(sp)
        # print(word, sp)
        sample[word].append(len(soundfile.read(os.path.join(path, df.iloc[i]['file']))[0]))
    data = pd.DataFrame(sample)
    from IPython import embed
    embed()