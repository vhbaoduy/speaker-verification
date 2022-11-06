import json
import pandas as pd

if __name__ == '__main__':
    info_1 = json.load(open('./gg-speech-v0.2/digits/filter_5.json'))
    info_2 = json.load(open('./gg-speech-v0.2/IOT/filter_5.json'))

    speaker = set(info_1['speakers']) & set(info_2['speakers'])
    print(len(speaker))

    df = pd.read_csv('./gg-speech-v0.2/data.csv')

    words = info_1['words'] + info_2['words']
    print(words)
    cnt = 0
    speakers = set()
    for word in words:
        for sp in speaker:
            temp = df[(df['word'] == word) & (df['speaker'] == sp)].count()
            if temp['file_name'] >= 5:
                # if cnt == 0:
                #     speakers = set(temp.index.to_list())
                # else:
                #     speakers = speakers & set(temp.index.to_list())
                speakers.add(sp)
                cnt+=1
    print(cnt)
    print(len(speakers))