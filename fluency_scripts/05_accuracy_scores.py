import argparse
import pandas as pd
import os
import glob
import numpy as np

def run(args):
    print('Start script 05 accuracy scores')
    basePath = args.asrDir
    asrSystem = args.asrSettings
    outputDir = args.outputDir

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
        
    fileList = glob.glob(os.path.join(basePath, asrSystem + '/csv-align-forward/' + '*.csv' ))

    if len(fileList) > 0:
    
        outputDictList = []
        for file in fileList:
            basename = os.path.basename(file).replace('.csv', '')

            outputDict = {}
            
            # Read DF and rename columns
            # df = pd.read_csv(file, index_col=0).rename(columns = {'prompt_label' : 'label', 'prompt_conf': 'start', 'prompt_start': 'end', 'prompt_end': 'conf', 'prompt_miscue': 'miscue'})
            df = pd.read_csv(file, index_col=0).rename(columns = {'prompt_label' : 'label', 'prompt_start': 'start', 'prompt_end': 'end', 'prompt_conf': 'conf', 'prompt_miscue': 'miscue'})

            # Get nr of correct and incorrect prompts
            try:
                nr_correct = df['correct'].value_counts()[True]
            except:
                nr_correct = 0 

            try:
                nr_incorrect = df['correct'].value_counts()[False]
            except:
                nr_incorrect = 0

            # Get total duration from ASR alignment file
            start_time = [x for x in df['start'] if x > 0.0][0]
            end_time = [x for x in df['end'] if x > 0.0][-1]

            duration_sec = end_time-start_time
            duration_min = duration_sec/60

            if duration_min == 0:
                print(basename)

            nrPrompts = len(df)
            outputDict['nr_correct'] = nr_correct
            outputDict['nr_incorrect'] = nr_incorrect
            outputDict['nr_prompts'] = nrPrompts
            outputDict['dur_sec'] = duration_sec
            outputDict['dur_min'] = duration_min
            outputDict['wcpm'] = nr_correct/duration_min
            outputDict['perc_cor'] = nr_correct/nrPrompts

            # get nr of deletions, substitutions, correct readings and other
            miscue_types = df['miscue'].value_counts().keys()
            for miscue_type in miscue_types:
                if miscue_type in ['del', 'sub', 'cor']: 
                    outputDict[miscue_type] = df['miscue'].value_counts()[miscue_type]
                else:
                    outputDict['other'] = len(df[~df['miscue'].isin(['del', 'sub', 'cor'])])

            # Read insertion file and get nr of insertions
            ins_file = file.replace('reversed', 'reversed-ins').replace('forward', 'forward-ins')
            insDF = pd.read_csv(ins_file, index_col=0)

            outputDict['ins'] = len(insDF)

            outputDictList.append({basename: outputDict})

        outputDF = pd.concat([pd.DataFrame.from_dict(x, orient='index') for x in outputDictList])

        outputDF['sub_perc'] = (outputDF['sub']/outputDF['nr_prompts']).round(4)
        outputDF['del_perc'] = (outputDF['del']/outputDF['nr_prompts']).round(4)
        outputDF['ins_perc'] = (outputDF['ins']/outputDF['nr_prompts']).round(4)
        outputDF['cor_perc'] = (outputDF['cor']/outputDF['nr_prompts']).round(4) 
        outputDF['cor_prompt_perc'] = (outputDF['nr_correct']/outputDF['nr_prompts']).round(4)

        # Fill nan values with zeroes is applicable
        for miscue_type in ['del', 'sub', 'cor', 'ins']:
            outputDF[miscue_type] = outputDF[miscue_type].fillna(0)
            outputDF[miscue_type + '_perc'] = outputDF[miscue_type + '_perc'].fillna(0)
        
        # Write outputDF
        outputFile = os.path.join(outputDir, 'accuracy.tsv')
        outputDF.round(3).sort_index().to_csv(outputFile, sep='\t')

        print('Finished script 05 accuracy scores: ', outputDir)

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