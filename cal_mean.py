import pandas as pd
import glob

if __name__ == '__main__':
    files = glob.glob('./stage1/exp*/result_noise.csv')
    df = pd.DataFrame()
    for file in files:
        df = pd.concat([pd.read_csv(file),df], ignore_index=True)
    from IPython import embed
    embed()