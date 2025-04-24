import pandas as pd
import os
import glob
import numpy as np
import argparse

def getDescriptiveStatistics(scores, dur_min):
    scores_dict = pd.Series(scores).describe().to_dict()
    try:
        scores_dict['IQR'] = round(scores_dict['75%'] - scores_dict['25%'],3)
    except:
        scores_dict['IQR'] = np.nan

    scores_dict['count_per_min'] = scores_dict['count'] / dur_min
    scores_dict['dur_min'] = dur_min
    return scores_dict

def getIntraWordPauses(sentenceDF):

    if len(sentenceDF)>0:

        pauses_end = list(sentenceDF['start'])[1:] # remove first start time, which represents end of initial pause
        pauses_start = list(sentenceDF['end'])[:-1] # remove last end time, which represents start of final pause
        
        # Get durations of pauses
        pauses_durations = np.array(pauses_end)-np.array(pauses_start)

        # Remove pauses with a duration < 0.2s (=200ms)
        pauses_durations_without0 = [pause_dur for pause_dur in pauses_durations if pause_dur >= 0.2]

        return [round(x,4) for x in pauses_durations_without0 if x > 0.0]
    
    else:
        return []

def run(args):

    print('Start script 06 inter intra pauses')

    basePath = args.asrDir
    asrSystem = args.asrSettings
    outputDir = args.outputDir

    # Create output dir inter and intra pauses
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # Create output dir sentences-with-timestamps
    outputDirSentenceTimestamps = os.path.join(basePath, asrSystem + '/timestamps-sentences')
    if not os.path.exists(outputDirSentenceTimestamps):
        os.makedirs(outputDirSentenceTimestamps)
        
    fileList = glob.glob(os.path.join(basePath, asrSystem + '/csv-align-forward/*.csv' ))

    if len(fileList) > 0:
    
        allDictList = []
        interDictList = []
        intraDictList = []
        for file in fileList:
            basename = os.path.basename(file).replace('.csv', '')
            
            # Read DF and rename columns
            # df = pd.read_csv(file, index_col=0).rename(columns = {'prompt_label' : 'label', 'prompt_conf': 'start', 'prompt_start': 'end', 'prompt_end': 'conf', 'prompt_miscue': 'miscue'})
            df = pd.read_csv(file, index_col=0).rename(columns = {'prompt_label' : 'label', 'prompt_start': 'start', 'prompt_end': 'end', 'prompt_conf': 'conf', 'prompt_miscue': 'miscue'})

            df['sentence_nr'] = [x.split('-')[0] for x in df.index]
            # print('sentence_nr:', sorted(list(set([int(x.split('-')[0]) for x in df.index]))))

            start_time = [x for x in df['start'] if x > 0.0][0]
            end_time = [x for x in df['end'] if x > 0.0][-1]
            dur_min = (end_time-start_time)/60

            ##############################
            ###     All Pauses         ###
            ##############################
            all_pauses = getIntraWordPauses(df[df['end'] > 0])
            # print(len(all_pauses), all_pauses)
            outputAllDict = getDescriptiveStatistics(all_pauses, dur_min)
            allDictList.append(pd.DataFrame.from_dict({basename : outputAllDict}))

            ##############################
            ### Intrasentential Pauses ###
            ##############################

            intrasentential_pause_list = []
            sentenceTimeList = []
            for sentence_nr in sorted(list(set([int(x.split('-')[0]) for x in df.index]))):
                
                # Get rows of DF with selected sentence number
                sentenceDF_intra = df[df['sentence_nr'] == str(sentence_nr)]
                sentenceDF_intra = sentenceDF_intra[sentenceDF_intra['end'] > 0].reset_index()
                if(len(sentenceDF_intra)>0):
                    begin_time_sentence = sentenceDF_intra.loc[0,'start']
                    end_time_sentence = sentenceDF_intra.loc[len(sentenceDF_intra)-1,'end']

                    # Add sentence with begin time and end time to sentenceTimeList
                    sentenceTimeList.append([begin_time_sentence, end_time_sentence, str(sentence_nr), " ".join(list(sentenceDF_intra.loc[0:len(sentenceDF_intra)-1,'label'].fillna('')))])

                    # Compute intrasentential pauses
                    intrasentential_pause_list.append(getIntraWordPauses(sentenceDF_intra[sentenceDF_intra['end'] > 0]))

            # Compute statistics from intrasentential pauses
            intrasentential_pauses = [item for sublist in intrasentential_pause_list for item in sublist]
            # print(len(intrasentential_pauses), intrasentential_pauses)
            outputIntraDict = getDescriptiveStatistics(intrasentential_pauses, dur_min)
            intraDictList.append(pd.DataFrame.from_dict({basename : outputIntraDict}))

            # Create list of sentences + start time and end time
            sentenceTimeDF = pd.DataFrame(sentenceTimeList, columns=['start', 'end', 'sentence_nr', 'sentence'])
            sentenceTimeDF.to_csv(os.path.join(outputDirSentenceTimestamps, basename + '.tsv'), sep = '\t')

            ##############################
            ### Intersentential Pauses ###
            ##############################
            intersentential_pauses = getIntraWordPauses(sentenceTimeDF)
            # print(len(intersentential_pauses), intersentential_pauses)
            outputInterDict = getDescriptiveStatistics(intersentential_pauses, dur_min)
            interDictList.append(pd.DataFrame.from_dict({basename : outputInterDict}))
            

            # # Create sentence-word identifier
            # df['sentence-word-idx'] = [x.split('-')[0] + '-' + x.split('-')[1] for x in df.index]

            # # Select only first and last word of each sentence
            # first_last_word_id_list= []
            # for sentence_nr in sorted(list(set([int(x.split('-')[0]) for x in df.index]))):
                
            #     # Select all words from a specific sentence
            #     sentenceDF = df[df['sentence_nr'] == str(sentence_nr)].reset_index()

            #     # Remove all word from sentenceDF that are not read (indicated by end == 0.0)
            #     sentenceDF = sentenceDF[sentenceDF['end'] > 0]

            #     # Add the begin and end sentence-word-idx to the first_last_word_id_list
            #     if len(sentenceDF) > 0:

            #         # first_word_id = 0
            #         # last_word_id = len(sentenceDF) - 1

            #         # first_last_word_id_list.append(str(sentence_nr) + '-' + str(first_word_id))
            #         # first_last_word_id_list.append(str(sentence_nr) + '-' + str(last_word_id))

            #         first_last_word_id_list.append('-'.join(sentenceDF.iloc[0,0].split('-', 2)[:2]))
            #         first_last_word_id_list.append('-'.join(sentenceDF.iloc[len(sentenceDF)-1,0].split('-', 2)[:2]))

            # # Select all rows from the first_last_word_id_list
            # interPauseDF = df[df['sentence-word-idx'].isin(first_last_word_id_list)]
            # intersentential_pauses = getIntraWordPauses(interPauseDF)[1::2]
            # print(len(intersentential_pauses), intersentential_pauses)
            # outputInterDict = getDescriptiveStatistics(intersentential_pauses, dur_min)
            # interDictList.append(pd.DataFrame.from_dict({basename : outputInterDict}))

        allDF = pd.concat(allDictList, axis=1).transpose().sort_index()
        allDF.columns = ['all_' + name for name in allDF.columns]
        allDF.round(3).sort_index().to_csv(os.path.join(outputDir, 'timing_all.tsv'), sep='\t')

        intraDF = pd.concat(intraDictList, axis=1).transpose().sort_index()
        intraDF.columns = ['intra_' + name for name in intraDF.columns]
        intraDF.round(3).sort_index().to_csv(os.path.join(outputDir, 'timing_intra.tsv'), sep='\t')

        interDF = pd.concat(interDictList, axis=1).transpose().sort_index()
        interDF.columns = ['inter_' + name for name in interDF.columns]
        interDF.round(3).sort_index().to_csv(os.path.join(outputDir, 'timing_inter.tsv' ), sep='\t')

        print('Created: ', os.path.join(outputDir, 'timing_all.tsv' ))
        print('Created: ', os.path.join(outputDir, 'timing_intra.tsv' ))
        print('Created: ', os.path.join(outputDir, 'timing_inter.tsv' ))
        print('Created: ', os.path.join(outputDirSentenceTimestamps, '*.tsv' ))
        
    
    else:
        print('No files in ./csv-align-forward')

def main():
    parser = argparse.ArgumentParser("Message")
    parser.add_argument("--asrDir", type=str, help = "Path to /04_asr dir.")
    parser.add_argument("--asrSettings", type=str, help = "ASR type")
    parser.add_argument("--outputDir", type=str, help = "outputDir")

    parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()