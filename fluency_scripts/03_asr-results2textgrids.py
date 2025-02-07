"""
This 
"""

import glob
import pandas as pd
import os
import json
import numpy as np
import tgt
import librosa
import argparse
import unidecode

import sclite.sclite_string_normalizer as sclite_norm # see dartastla package

# normalized_string = sclite_norm.normalize_string('Hallo! Ik*u ben van-vandaag hier*a met 50 en 1 persoon s\'avonds, waaronder m\'n Cas en Lucas. Hoe is \'t?', annTags=True)
# print('\n', normalized_string)

# punc = '''()-[]{};:,\<>/@#$%^&*_~''' #'''()-[]{};:,'"\<>/@#$%^&*_‘~’'''

# def normalizeNumbers(txt):
#     return txt.replace('30', 'dertig').replace('13', 'dertien').replace('2', 'twee')

# def normalizeJojo(txt):

#     for jojo_variant in ['yoyo', 'yo-yo']:
#         txt = txt.replace(jojo_variant, 'jojo').lower()

#     for jojo_variant in ['yoyos', 'yo-yos', 'yoyo\'s', 'yo-yo\'s']:
#         txt = txt.replace(jojo_variant, 'jojo\'s').lower()
            
#     return txt

# def removePunctuation(s):
#     return "".join([letter for letter in s if letter not in punc])

def readWhisperToutputJSON(jsonFile):
    with open(jsonFile, 'r') as f:
        data = json.load(f)
    return data

def obj2interval(obj):
    start = obj['start']
    end = obj['end']
    txt = obj['text']

    return tgt.core.Interval(start, end, text=txt)

def obj2dfRow(obj):
    start = obj['start']
    end = obj['end']
    txt = obj['text']
    txt_norm = sclite_norm.normalize_string(txt)

    return [txt, start, end, txt_norm]

def obj2intervalSegm(obj):
    start = obj['start']
    end = obj['end']
    txt = obj['text'].strip()

    txt = sclite_norm.normalize_string(txt)

    return tgt.core.Interval(start, end, text=txt)

def obj2intervalConf(obj):
    start = obj['start']
    end = obj['end']
    txt = str(obj['confidence'])

    return tgt.core.Interval(start, end, text=txt)


def run(args):

    print('Start:', os.path.basename(__file__))

    jsonAsrResultsDir = args.jsonAsrResultsDir
    audioDir = args.audioDir

    # List input files
    jsonAsrResultsList = glob.glob(os.path.join(jsonAsrResultsDir, '*.json'))
    print(len(jsonAsrResultsList), 'json ASR results are found.')

    # Create output directory for TextGrids
    outputDir = jsonAsrResultsDir.replace('json-asr-results', 'json-asr-results-as-tg')
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # Create output directory for csv with timestamped recognized words
    outputDirDF = jsonAsrResultsDir.replace('json-asr-results', 'timestamps-words')
    if not os.path.exists(outputDirDF):
        os.makedirs(outputDirDF)

    # Create list to fill with ASR transcriptions 
    asrTransList_orig = []
    asrTransList_norm = []

    # Convert each json file to a TextGrid file
    for jsonAsrResult in jsonAsrResultsList:

        # Get basename of file
        basename = os.path.basename(jsonAsrResult).replace('.json', '')

        # Read JSON file
        data = readWhisperToutputJSON(jsonAsrResult)

        # Extract duration of corresponding audio file
        audioPath = os.path.join(audioDir, basename + '.wav')
        y, sr = librosa.load(audioPath, sr=16000)
        durLibrosa = librosa.get_duration(y=y, sr=sr)

        # Extract text from json file and save this original transcription
        asrTransList_orig.append([basename, data['text']])

        # Extract segments from JSON file
        segments = data['segments']

        wordItemsList = []

        try:

            # Create Segment Tier
            segments_intervals = [obj2intervalSegm(segment) for segment in segments]
            segmentsTier = tgt.core.IntervalTier(start_time=0.0, end_time=durLibrosa, name='segments', objects=None)
            segmentsTier.add_intervals(segments_intervals)

            # Create Words Tier (with disfluecies)
            items = [segment['words'] for segment in segments]
            items_flatten = [word for words_segment in items for word in words_segment]
            items_flatten_intervals = [obj2interval(obj) for obj in items_flatten]
            wordsDisTier = tgt.core.IntervalTier(start_time=0.0, end_time=durLibrosa, name='wordsDis', objects=None)
            wordsDisTier.add_intervals(items_flatten_intervals)

            # Create Words Tier (without disfluecies)
            words = [item for item in items_flatten if item['text'] != "[*]"]
            words_intervals = [obj2interval(obj) for obj in words]
            wordsTier = tgt.core.IntervalTier(start_time=0.0, end_time=durLibrosa, name='words', objects=None)
            wordsTier.add_intervals(words_intervals)

            # Create DataFrame of Words Tier
            word_items = [obj2dfRow(obj) for obj in words]
            wordItemsList.append(word_items)

            # Create words confidence score tier
            conf_intervals = [obj2intervalConf(obj) for obj in words]
            confTier = tgt.core.IntervalTier(start_time=0.0, end_time=durLibrosa, name='conf', objects=None)
            confTier.add_intervals(conf_intervals)

            # Add all tiers to TextGrid
            tg = tgt.core.TextGrid()
            tg.add_tier(wordsDisTier)
            tg.add_tier(wordsTier)
            tg.add_tier(confTier)
            tg.add_tier(segmentsTier)

            # Write TextGrid
            outputFile = os.path.join(outputDir, basename + '.TextGrid')
            tgt.io.write_to_file(tg, outputFile, format='long', encoding='utf-8')

        except:

            print("Creating TextGrid not possible:", basename)

        # Create 'asr-words-with-timestamps' files
        transposedWordItemsList = np.transpose(wordItemsList)
        transposedWordItemsList_flatten = [[item for sublist in l for item in sublist] for l in transposedWordItemsList]
        df = pd.DataFrame(transposedWordItemsList_flatten).transpose()
        df.columns = ['text', 'start', 'end', 'text_norm']
        df.to_csv(os.path.join(outputDirDF, basename + '.tsv'), sep='\t')

        # Add ASR transcription to list with ASR transcriptions
        asrTransList_norm.append([basename, " ".join(list(df['text_norm'])).strip()])

    asrTransDF_orig = pd.DataFrame(asrTransList_orig, columns = ['audioID','ASR_transcription']).set_index('audioID').sort_index()
    asrTransDF_orig.to_csv(jsonAsrResultsDir.replace('json-asr-results', 'at-orig.csv'))

    asrTransDF_norm = pd.DataFrame(asrTransList_norm, columns = ['audioID','ASR_transcription']).set_index('audioID').sort_index()
    asrTransDF_norm.to_csv(jsonAsrResultsDir.replace('json-asr-results', 'at-norm.csv'))

    print('The following files are created:')
    print(len(glob.glob(os.path.join(outputDirDF, '*.tsv'))), 'tsv files in' , outputDirDF)
    print(len(glob.glob(os.path.join(outputDir, '*.TextGrid'))), 'TextGrid files in' , outputDir)
    print('The file', jsonAsrResultsDir.replace('json-asr-results', 'at-orig.csv'))
    print('The file', jsonAsrResultsDir.replace('json-asr-results', 'at-norm.csv'))


def main():
    parser = argparse.ArgumentParser("Message")
    parser.add_argument("--jsonAsrResultsDir", type=str, help = "Path to json-asr-results directory.")
    parser.add_argument("--audioDir", type=str, help = "Path to audio directory.")

    parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()