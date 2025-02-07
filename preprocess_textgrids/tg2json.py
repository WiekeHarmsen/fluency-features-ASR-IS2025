import pandas as pd
import os
import json
import glob

from utils import read_textgrids as rf
import sclite.sclite_string_normalizer as sclite_norm

def readTextGridFile(tgFile, corpus):
    # Read TextGrid file
    if corpus == 'serda':
        tg_df = rf.read_tg_file_to_df(tgFile, 'latin-1')
    elif corpus == 'jasmin':
        tg_df = rf.read_tg_file_to_df_jasmin(tgFile)
    return tg_df.astype({'start_time':float, 'end_time':float} )

def selectWordTierTextGrid(tg_df, word_tier_name):
    return tg_df[tg_df['tier_name'] == word_tier_name]

def splitTextDFIntoSentences(tg_df_orth_trans):
    sentenceDFList = []
    startIDX = tg_df_orth_trans.index[0]
    for idx, row in tg_df_orth_trans.iterrows():
        if row['text'][-1] in ['.', '!', '?']:
            sentenceDFList.append(tg_df_orth_trans.loc[startIDX:idx, :])
            startIDX = idx+1
    return sentenceDFList

def wordRowToWordSegment(row):
    # Remove annotation tags (*u, *a, etc.), remove all punctuation except the basic punctuation (!-'.?) and all default normalization steps (poss. pronouns, names, spelling errors, write numbers as words)
    w = sclite_norm.normalize_string(row['text'], annTags=True, all_punct=False, basic_punct=True)
    if ' ' in w:
        print(w)

    return {
                "text": w.replace(' ', ''),
                "start": row['start_time'],
                "end": row['end_time'],
                "confidence": 0.0
            }


def turnSentenceDFIntoSegment(sentenceDF, sentenceNr):

    # Remove _ words, these are noisy areas before/after words
    sentenceDF = sentenceDF[sentenceDF['text'] != '_']

    wordsList = list(sentenceDF.apply(wordRowToWordSegment, axis=1))

    return {
            "id": sentenceNr,
            "seek": 0,
            "start": sentenceDF.loc[sentenceDF.index[0], 'start_time'],
            "end": sentenceDF.loc[sentenceDF.index[-1], 'end_time'],
            "text": " ".join([x['text'] for x in wordsList]),
            "words": wordsList
    }


def main():
    parser = argparse.ArgumentParser("Message")
    parser.add_argument("--jsonAsrResultsDir", type=str, help = "Path to json-asr-results directory.")
    parser.add_argument("--outputFile", type=str, help = "Path to dir where the output file should be saved.")

    parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
