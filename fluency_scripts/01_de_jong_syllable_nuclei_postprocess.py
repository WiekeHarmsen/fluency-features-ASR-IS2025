import argparse
import glob
import pandas as pd
import os
import numpy as np

def normalizeMissingValues(v):
    if v == '--undefined--' or v == '':
        return np.nan
    else:
        return v


def run(args):

    print('01_de_jong_syllable_nuclei_postprocess.py started')

    fluencyFeatureTxt = args.fluencyFeatureTxt
    fluencyFeatureTsv = args.fluencyFeatureTsv

    print('Input txt:', fluencyFeatureTxt)
    print('Input tsv:', fluencyFeatureTsv)

    with open(fluencyFeatureTxt, 'r') as f:
        data = [x.replace('\n', '') for x in f.readlines()]

    # Remove empty lines from data
    data = [line for line in data if line != '']

    outputMatrix = []
    for idx, line in enumerate(data):
        modulo_idx = idx%3
        # modulo_idx == 0 : path to audio file
        # modulo_idx == 1 : header
        # modulo_idx == 2 : computed measures

        if modulo_idx == 2:
            outputMatrix.append(normalizeMissingValues(item) for item in line.split(', '))

    header = data[1].split(', ')
    df = pd.DataFrame(outputMatrix, columns = header)
    
    # Set index to <speaker>-<task>
    df['audioID'] = df['name'].apply(lambda x: x.split('-')[0] + '-'+ x.split('-')[1])
    df = df.drop('name', axis=1).set_index('audioID')
    df.to_csv(fluencyFeatureTsv, sep='\t', na_rep = 'NA')

    print('Output tsv:', fluencyFeatureTsv)
    print('01_de_jong_syllable_nuclei_postprocess.py finished')
    

def main():
    parser = argparse.ArgumentParser("Message")
    parser.add_argument("--fluencyFeatureTxt", type=str, help = "Fluency feature Txt file (output of de_jon_syllable_nuclei_v3.py)")
    parser.add_argument("--fluencyFeatureTsv", type=str, help = "Fluency feature Tsv file (to which txt is converted)")

    parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()