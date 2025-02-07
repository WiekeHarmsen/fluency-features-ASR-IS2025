import glob
import pandas as pd
import os
import json
import numpy as np
import argparse

def renameFile(fileName):
    spk = fileName.split('-')[0]
    task = fileName.split('-')[1]
    return spk + '-' + task.replace('.json', '')

def readWhisperToutputJSON(jsonFile):
    with open(jsonFile, 'r') as f:
        data = json.load(f)
    return data

def getDescriptiveStatistics(scores):
    scores_dict = pd.Series(scores).describe().to_dict()
    try:
        scores_dict['IQR'] = round(scores_dict['75%'] - scores_dict['25%'],3)
    except:
        scores_dict['IQR'] = np.nan
    return scores_dict

def itemDurationAndConfidenceAnalysis(items):
    if len(items) > 0:
        durations = [round(item['end']-item['start'],3) for item in items]
        stats_durations = getDescriptiveStatistics(durations)

        conf = [item['confidence'] for item in items]
        stats_conf = getDescriptiveStatistics(conf)
    else:
        stats_durations = getDescriptiveStatistics([np.nan,np.nan,np.nan,np.nan])
        stats_conf = getDescriptiveStatistics([np.nan,np.nan,np.nan,np.nan])

    return stats_durations, stats_conf

def pausesAnalysis(items):

    if len(items)>0:

        pauses_end = [item['start'] for item in items][1:] # remove first start time, which represents end of initial pause
        pauses_start = [item['end'] for item in items][:-1] # remove last end time, which represents start of final pause
        
        # Get durations of pauses
        pauses_durations = np.array(pauses_end)-np.array(pauses_start)

        # Remove pauses with a duration < 0.2s (=200ms)
        pauses_durations_without0 = [pause_dur for pause_dur in pauses_durations if pause_dur >= 0.2]

        # Compute statistic measures of pauses
        stats_pauses_durations = getDescriptiveStatistics(pauses_durations_without0)
    
    else:
        stats_pauses_durations = getDescriptiveStatistics([np.nan,np.nan,np.nan,np.nan])
        pauses_durations = getDescriptiveStatistics([np.nan,np.nan,np.nan,np.nan])

    return stats_pauses_durations, pauses_durations
    

def changeNamesOfKeys(outputDict, prefix):
    return dict((prefix+key, value) for (key, value) in outputDict.items())

def getReadingFluencyStatistics(words, pauses_durations, pauses2_durations, nrOfSegments):

    if len(words)>0:

        # Total duration (excl. begin and end silence)
        startTimeReading = [word['start'] for word in words][0]
        endTimeReading = [word['end'] for word in words][-1]
        totalReadingTime = endTimeReading - startTimeReading
        speechRate = round(len(words)/(totalReadingTime/60), 3)
        phonationTime = totalReadingTime - pauses_durations.sum()
        phonationTime2 = totalReadingTime - pauses2_durations.sum()
        articulationRate = round(len(words)/(phonationTime/60), 3)
        articulationRate2 = round(len(words)/(phonationTime2/60), 3)
    
    else:
        totalReadingTime, speechRate, phonationTime, phonationTime2, articulationRate, articulationRate2, nrOfSegments = [np.nan] * 7


    return {
        'totalReadingTime' : totalReadingTime,
        'speechRate(WPM)' : speechRate,
        'phonationTime' : phonationTime,
        'phonationTime2' : phonationTime2,
        'articulationRate' : articulationRate,
        'articulationRate2' : articulationRate2,
        'nrOfSegments' : nrOfSegments,
    }



def run(args):

    jsonAsrResultsDir = args.jsonAsrResultsDir
    outputFile = args.outputFile

    if not os.path.exists(os.path.dirname(outputFile)):
        os.makedirs(os.path.dirname(outputFile))

    jsonAsrResultsList = glob.glob(os.path.join(jsonAsrResultsDir, '*.json'))

    outputDict = {}

    for jsonAsrResult in jsonAsrResultsList:

        # Get basename of file
        basename = renameFile(os.path.basename(jsonAsrResult))

        # 1. Read JSON file
        data = readWhisperToutputJSON(jsonAsrResult)

        # 2. Extract features from JSON file
        text = data['text']
        segments = data['segments']

        # SEGMENTS
        nrOfSegments = len(segments)

        try:

            # ITEMS
            # Extract items, these can either be recognized words or disfluencies [*]
            items = [segment['words'] for segment in segments]
            items_flatten = [word for words_segment in items for word in words_segment]
            disfluencies = [item for item in items_flatten if item['text'] == "[*]"]
            words = [item for item in items_flatten if item['text'] != "[*]"]

            # ITEMS - DISFLUENCIES
            stats_durations_disfluencies, stats_conf_disfluencies = itemDurationAndConfidenceAnalysis(disfluencies)
            stats_durations_disfluencies = changeNamesOfKeys(stats_durations_disfluencies, 'disfl_dur_')
            stats_conf_disfluencies = changeNamesOfKeys(stats_conf_disfluencies, 'disfl_conf_')
            
            # ITEMS - WORDS
            stats_durations_words, stats_conf_words = itemDurationAndConfidenceAnalysis(words)
            stats_durations_words = changeNamesOfKeys(stats_durations_words, 'words_dur_')
            stats_conf_words = changeNamesOfKeys(stats_conf_words, 'words_conf_')

            # ITEMS - PAUZES I (disfluencies are pauses)
            stats_pauses_durations, pauses_durations = pausesAnalysis(words)
            stats_pauses_durations = changeNamesOfKeys(stats_pauses_durations, 'pauses_dur_')

            # ITEMS - PAUZES II (disfluencies are not pauses)
            stats_pauses2_durations, pauses2_durations = pausesAnalysis(items_flatten)
            stats_pauses2_durations = changeNamesOfKeys(stats_pauses2_durations, 'pauses2_dur_')

            # Overall reading fluency statistics
            stats_reading_fluency = getReadingFluencyStatistics(words, pauses_durations, pauses2_durations, nrOfSegments)

            outputDict[basename] = {**stats_reading_fluency, **stats_durations_disfluencies, **stats_conf_disfluencies, **stats_durations_words, **stats_conf_words, **stats_pauses_durations, **stats_pauses2_durations}

        except Exception as error:
            print('not possible:', basename, error)
    
    df = pd.DataFrame(outputDict).transpose().round(3)
    df.to_csv(outputFile, sep='\t')

def main():
    parser = argparse.ArgumentParser("Message")
    parser.add_argument("--jsonAsrResultsDir", type=str, help = "Path to json-asr-results directory.")
    parser.add_argument("--outputFile", type=str, help = "Path to dir where the output file should be saved.")

    parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

