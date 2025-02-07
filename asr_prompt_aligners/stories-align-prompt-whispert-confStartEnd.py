"""
This script takes as input one ASR_result (whisperT, .json) and corresponding prompt file (.prompt).
This script reads the files, extracts the relevant information, aligns the prompt and ASR transcription and outputs this.
The relevant information from the prompt file is only the reference transcription (what the child should read).
The relevant information from the ASR result file is the 'segments' property. 
This is a dictionary with as value an object with the following word properties: label, start_time, end_time and confidence score.

This script is an improved version of /vol/tensusers5/wharmsen/ASTLA/astla-round1/2-story2file-info.ipynb
and /vol/tensusers5/wharmsen/ASTLA/astla-round1/4-add-story-conf-info.ipynb

OUTPUT: 
A directory with one or multiple .csv files with an alignment of whisper output and prompt.

Example:
promptID    aligned_asrTrans    reversed_aligned_asrTrans   correct confidence  startTime   endTimes
0-0-Bang    *a*l                *als                        False   0.0         0.0         0.0
"""

import pandas as pd
import glob
import os
import json
import re
import argparse
from datetime import datetime
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning) # Blocks pandas loc/iloc errors
import numpy as np

# Local package 'dartastla'
# First install this package by
# cd /vol/tensusers5/wharmsen/dartastla
# pip install -e .
# python -c "import sys; print(sys.path)"
# import alignment_adagt.string_manipulations as strman
import sclite.sclite_string_normalizer as sclite_norm
import alignment_modern as alignmod

# nohup time python ./02-stories-align-prompt-whispert.py &

"""
This function reads one .prompt file and normalizes the text (trim spaces, remove accents, remove punctuation, remove digits)
"""
def readPromptFile(path_to_prompt_file):

    with open(path_to_prompt_file, 'r') as f:
        promptRaw = f.read().replace('\n', ' ')

    prompt = sclite_norm.normalize_string(promptRaw, annTags=False)
    return prompt

def getPromptIdxs(promptIdxFile):

    promptDF = pd.read_csv(promptIdxFile)

    return list(promptDF['prompt_id'])

"""
This function reads one json file with an WhisperT AsrResult.
The asrTranscription is normalized (trim spaces, remove accents, remove punctuation, remove digits)
"""
def readAsrResult(asrResultFile):

    # Read json file
    with open(asrResultFile, 'r') as f:
        asrResult = json.load(f)

    # Get 'text' property
    asrTranscriptionRaw = asrResult['text']
    asrTranscription = sclite_norm.normalize_string(asrTranscriptionRaw)

    # Get 'segments' property
    recWordsList = []
    for segment in asrResult['segments']:
        words = segment['words']
        for word in words:

            label = sclite_norm.normalize_string(word['text'])

            # If no disfluency
            if (label != '[*]' and label != ''):

                start = word['start']
                end = word['end']
                confidence = word['confidence']

                recWordsList.append([label, start, end, confidence])

    recWordsDF =  pd.DataFrame(recWordsList, columns= ['label', 'confidence', 'start', 'end'])    

    return asrTranscription, recWordsDF


"""
This is a recursive function. The dynamic alignment algorithm ADAGT doesn't work on long texts. 
Therefore, we find a piece of text that occurs in both transcriptions, and split the long text into shorter texts at this piece of text.
This is a recursive function to make this splitting happen.
The properties targetPartsList and origPartsList are in the end used for further analysis.

targetPartsList     string[]:   The target_text splitted in utterances
origPartsList       string[]:   The original_text splitted in utterances
target_text         string:     The complete prompt (can be unlimited words).
original_text       string:     The reading of the prompt as recognized by the ASR.
original_space_idx  int:    The idx at which a split can be made
max_length          int:    Length of utterance.
"""
def makeSplit(targetPartsList, origPartsList, target_text, original_text, original_space_idx, max_length):

    # Get slice to split on
    original_slice = original_text[original_space_idx-3:original_space_idx+3]

    # Find slice in target text
    target_slice_idx = target_text.find(original_slice)

    # If slice is found in target text
    if (target_slice_idx != -1 and target_slice_idx < original_space_idx+200 and len(original_slice) != 0):

        # Split the target_text and original_text on the space in the overlapping slice
        # Add the first part to the partLists
        target_space_idx = target_slice_idx+3
        targetPartsList.append(target_text[:target_space_idx])
        origPartsList.append(original_text[:original_space_idx])

        # Remove the first part from the target_text and original_text
        target_text = target_text[target_space_idx+1:]
        original_text = original_text[original_space_idx+1:]

        if (len(original_text) < 80 or len(target_text) < 80):
            # End of file reached
            targetPartsList.append(target_text)
            origPartsList.append(original_text)
            target_text = ''
            original_text = ''
            idx_of_next_space = -1

            return targetPartsList, origPartsList, target_text, original_text, idx_of_next_space, max_length
        else:
            # reset around_idx
            new_space_idx = original_text.find(" ", max_length)

            # make next split
            return makeSplit(targetPartsList, origPartsList, target_text, original_text, new_space_idx, max_length)

    # If slice is not found in target text
    else:
        # reset around_idx to next space
        idx_of_next_space = original_text.find(" ", original_space_idx+1)
        if (idx_of_next_space == -1):
            # End of file reached
            targetPartsList.append(target_text)
            origPartsList.append(original_text)
            return targetPartsList, origPartsList, target_text, original_text, idx_of_next_space, max_length
        else:
            return makeSplit(targetPartsList, origPartsList, target_text, original_text, idx_of_next_space, max_length)

"""
Function that checks whether the three input files exist, if not: print error message.
"""
def checkIfFilesExist(asrResultFile, promptFile):

    for file in [asrResultFile, promptFile]:
        if not os.path.exists(file):
            print(file, 'does not exist.')

"""
Function that reads the input files, calls the alignment algorithm and prints the output.

asrTranscription    string
prompt              string
"""
def alignOneFile(asrTranscription, prompt):
    promptAlignDF = alignmod.two_way_alignment_modern(prompt, asrTranscription)
    return promptAlignDF

"""
This function searches in asrResultWordsDict to the values that has as label "prompt". If there are multiple options, it chooses the one with the dict index directly

prompt              string: One prompt word
asrResultWordsDict  dict:   The key is the index, the values are {label: string, start_time: float, end_time: float, confidence: float} objects. 
                            This dict contains all recognized word segments from the ASR result.
indexThreshold      int:    Select the first value that has an the 
"""
def searchCorrespondingConfidence(prompt, asrResultWordsDict, indexThreshold):

    # Make list of all dict values with prompt as target_label.
    allDictValues = [val for val in asrResultWordsDict.values() if val['label'] == prompt]

    # Get indexes in asrResultsWordsDict of this selection of dict values
    allDictIdxs = [[x[0] for x in asrResultWordsDict.items() if x[1] == value][0] for value in allDictValues]

    # Select one of these dict values. Use the indexThreshold for that.
    currentDictIdx = -1
    for i, idxItem in enumerate(allDictIdxs):
        if (idxItem > indexThreshold):
            currentDictValue = allDictValues[i]
            currentDictIdx = idxItem
            break

    if (currentDictIdx == -1):
        # Means that prompt is part of recognized word (e.g prompt=rilt, recognized=trilt)
        confidence = 999
        start = 999
        end = 999
    else:
        confidence = asrResultWordsDict[currentDictIdx]['confidence']
        start = asrResultWordsDict[currentDictIdx]['start']
        end = asrResultWordsDict[currentDictIdx]['end']

    return currentDictIdx, {'confidence': confidence, 'start': start, 'end': end}

def findAllSpaceInsertions(aligned_ref, aligned_asrTrans):
    # Find all space insertions
    space_ins_char_list = []
    for char_idx in range(len(aligned_ref)):
        
        ref_char = aligned_ref[char_idx]
        asr_char = aligned_asrTrans[char_idx]
        
        # Find insertion of space
        if ref_char == '*' and asr_char == ' ':
            space_ins_char_list.append(char_idx)
    return space_ins_char_list

def splitRefAndAsrTransOnSpaceIns(space_ins_char_list, aligned_ref, aligned_asrTrans):
    w_hyp_list = []
    w_ref_list = []
    word_split_list_begin_idx = [0] + space_ins_char_list

    for i in range(len(word_split_list_begin_idx)):
        begin = word_split_list_begin_idx[i]

        if i != len(word_split_list_begin_idx)-1:
            end = word_split_list_begin_idx[i+1]
        else:
            end = len(aligned_ref)
        
        w_ref = aligned_ref[begin: end]
        w_hyp = aligned_asrTrans[begin: end]

        if w_hyp[0] == ' ':
            w_hyp = w_hyp[1:]
            w_ref = w_ref[1:]

        w_ref_list.append(w_ref)
        w_hyp_list.append(w_hyp)

    return w_ref_list, w_hyp_list

def addConfidenceScores(promptAlignDF, recWordsDF):

    promptAlignDF = promptAlignDF.reset_index()

    # convert recWordsDF to array 
    recWordsList = recWordsDF.values.tolist()

    insertionList = []
    confStartEndList = []
    for prid, row in promptAlignDF.iterrows():

        aligned_asrTrans_key = 'aligned_asrTrans'
        aligned_ref_key = 'aligned_ref'

        # Get original alignments for insertion detection
        asrTransOrig = row[aligned_asrTrans_key]
        refOrig = row[aligned_ref_key]

        # Select first aligned_asrTrans and remove *
        asrTrans = row[aligned_asrTrans_key].replace('*', '')

        # Select corresponding prompt 
        prompt = row[aligned_ref_key].replace('*', '')

        # Compute nr of words in asrTrans
        nrWordsAsrTrans = 0 if asrTrans == '' else len(asrTrans.split(' '))
        # print('asrTrans:', asrTrans.split(' '), nrWordsAsrTrans, ''.split(' '))
        # print('prompt:', prompt)
        # print('recWordsList[0][0]', recWordsList[0][0])
        # print('recWordsList[0][0] == asrTrans', recWordsList[0][0] == asrTrans)

        if nrWordsAsrTrans == 1:
            # Check if first entry in recWordList is equal to asrTrans or prompt
            # Yes? Select the first entry and remove it from asrWordInfoDict
            # No? Probably the prompt word is not read by the child. Treat it as a deleted/skipped word (all zeroes).
            if(recWordsList[0][0] == asrTrans or recWordsList[0][0] == prompt):
                # This is a substitution or correctly read word.

                # Extract relevant information
                prompt_label = recWordsList[0][0]
                prompt_conf = recWordsList[0][1]
                prompt_start = recWordsList[0][2]
                prompt_end = recWordsList[0][3]
                if asrTrans == prompt:
                    prompt_miscue = 'cor'
                else:
                    prompt_miscue = 'sub'
                
                # Update recWordsList
                recWordsList = recWordsList[1:]
            else:              

                # This is skipped (deleted) word
                prompt_label = ''
                prompt_conf = np.nan
                prompt_start = np.nan
                prompt_end = np.nan
                prompt_miscue = 'del'


                firstWords = [x.replace('*', '') for x in list(promptAlignDF[aligned_asrTrans_key])[prid:]]
                firstFiveNonEmptyWords = [x for x in firstWords if x != ''][:5]
                if not recWordsList[0][0] in firstFiveNonEmptyWords:
                    recWordsList = recWordsList[1:]
                
                
        elif nrWordsAsrTrans == 0:
            # The prompt word is not read by the child. Treat it as a deleted/skipped word (all zeroes).
            prompt_label = ''
            prompt_conf = 0
            prompt_start = 0
            prompt_end = 0
            prompt_miscue = 'del'

        elif nrWordsAsrTrans >= 2:

            if(recWordsList[0][0] == asrTrans or recWordsList[0][0] == prompt):
                # This statement is added to catch words like 's nachts'

                # Extract relevant information
                prompt_label = recWordsList[0][0]
                prompt_conf = recWordsList[0][1]
                prompt_start = recWordsList[0][2]
                prompt_end = recWordsList[0][3]
                if asrTrans == prompt:
                    prompt_miscue = 'cor'
                else:
                    prompt_miscue = 'sub'
                
                # Update recWordsList
                recWordsList = recWordsList[1:]

            else:

                space_ins_char_list = findAllSpaceInsertions(refOrig, asrTransOrig)
                refWordList, asrTransWordList = splitRefAndAsrTransOnSpaceIns(space_ins_char_list, refOrig, asrTransOrig)

                if len(refWordList) != len(asrTransWordList):
                    print('ERROR: unequal lengths: ', refWordList, asrTransWordList)

                correctFound = False
                subPromptList = []
                for idx, asrTransWord in enumerate(asrTransWordList):
                    prompt = refWordList[idx]

                    asrTransWord = asrTransWord.replace('*', '').strip()
                    prompt = prompt.replace('*', '').strip()

                    # Is asrTransWord the first entry in asrWordInfoDict?
                    # Yes? Select the first entry and remove it from asrWordInfoDict
                    # No? The word is probably inserted by the child. Treat it as an insertion.
                    if(recWordsList[0][0] == asrTransWord and recWordsList[0][0] == prompt):
                        # Extract relevant information
                        subprompt_label = recWordsList[0][0]
                        subprompt_conf = recWordsList[0][1]
                        subprompt_start = recWordsList[0][2]
                        subprompt_end = recWordsList[0][3]
                        subprompt_miscue = 'cor'
                        
                        # Update recWordsList
                        recWordsList = recWordsList[1:]

                        # Set correctFound to True
                        correctFound = True

                        subPromptList.append([subprompt_label, subprompt_conf, subprompt_start, subprompt_end, subprompt_miscue])

                    elif(recWordsList[0][0] == asrTransWord and recWordsList[0][0] != prompt):

                        if prompt == '':
                            # insertion
                            prompt_with_ins = prid
                            pos_ins_word = idx
                            ins_label = recWordsList[0][0]
                            ins_conf = recWordsList[0][1]
                            ins_start = recWordsList[0][2]
                            ins_end = recWordsList[0][3]
                            ins_miscue = 'ins'
                            cor_already_found = correctFound

                            insertionList.append([prompt_with_ins, pos_ins_word, ins_label, ins_conf, ins_start, ins_end, ins_miscue, cor_already_found])

                            # Update recWordsList
                            recWordsList = recWordsList[1:]

                        else:
                            # substitution
                            # Extract relevant information
                            subprompt_label = recWordsList[0][0]
                            subprompt_conf = recWordsList[0][1]
                            subprompt_start = recWordsList[0][2]
                            subprompt_end = recWordsList[0][3]
                            subprompt_miscue = 'sub'

                            subPromptList.append([subprompt_label, subprompt_conf, subprompt_start, subprompt_end, subprompt_miscue])
                            
                            # Update recWordsList
                            recWordsList = recWordsList[1:]
                    
                    elif(recWordsList[0][0].find(asrTransWord) != -1 and recWordsList[0][0] != prompt):
                        # The prompt word is not read by the child. Treat it as a deleted/skipped word (all zeroes).
                        subprompt_label = ''
                        subprompt_conf = 0
                        subprompt_start = 0
                        subprompt_end = 0
                        subprompt_miscue = 'del'

                        subPromptList.append([subprompt_label, subprompt_conf, subprompt_start, subprompt_end, subprompt_miscue])


                    else:
                        print('This should not happen, if it does: find out what to do...')
                        print('recWordsList[0][0]', recWordsList[0][0])
                        print('asrTransWord:', asrTransWord)
                        print(recWordsList[0][0] == asrTransWord)
                        print('prompt:', prompt)
                        print(recWordsList[0][0] == prompt)


            # print(pd.DataFrame(subPromptList))
            # Extract prompt info from subPromptList
            prompt_label = " ".join([x[0] for x in subPromptList])
            prompt_conf = np.mean([float(x[1]) for x in subPromptList]) # 1=subprompt_start
            prompt_start = subPromptList[0][2] # 2=subprompt_start
            prompt_end = subPromptList[-1][3] # 3=subprompt_end
            prompt_miscue = "-".join([x[4] for x in subPromptList]) # 4=reading miscue

        confStartEndList.append([prompt_label, prompt_conf, prompt_start, prompt_end, prompt_miscue])

    confStartEndDF = pd.DataFrame(confStartEndList, index=promptAlignDF.index, columns=['prompt_label', 'prompt_start', 'prompt_end', 'prompt_conf', 'prompt_miscue'])
    promptAlignDF = pd.concat([promptAlignDF, confStartEndDF], axis=1)

    insertionDF = pd.DataFrame(insertionList, columns = ['prompt_with_ins', 'pos_ins_word', 'ins_label', 'ins_conf', 'ins_start', 'ins_end', 'ins_miscue', 'cor_already_found'] )

    return promptAlignDF, insertionDF

def correctForNotReadSentences(promptAlignConfDF, insertionDF, asrTranscription, recWordsDF):
    # Check for each sentence whether it is read or not

    # Add sentenceNr to the promptAlignConfDF
    promptAlignConfDF['sentence_nr'] = [int(x.split('-')[0]) for x in promptAlignConfDF.index]

    # Get for each sentence the percentage of deleted words (prompt_miscue = del)
    unique_sentences = sorted(list(set(promptAlignConfDF['sentence_nr'])), reverse=True)

    # Loop through each sentence, start from the back. If percentage of del words > 50%, the sentence is probably not read by the child.
    id_last_read_sentence = unique_sentences[0]
    for sentence_idx in unique_sentences:
        sentenceDF = promptAlignConfDF[(promptAlignConfDF['sentence_nr'] == sentence_idx)]
        percDelPerSentence = len(sentenceDF[sentenceDF['prompt_miscue'] == 'del']) / len(sentenceDF)
        if(percDelPerSentence <= 0.5):
            id_last_read_sentence = sentence_idx
            break

    nr_missing_sentences = unique_sentences[0] - id_last_read_sentence
    perc_sentences_read = round((id_last_read_sentence + 1) /len(unique_sentences),3)

    sentenceStats = [id_last_read_sentence, nr_missing_sentences, perc_sentences_read, len(unique_sentences)]

    if id_last_read_sentence != unique_sentences[0]:

        # Detect the first sentence that is not read and split the prompt in two parts: read_prompts and not_read_prompts
        promptAlignConfDF_read = promptAlignConfDF[promptAlignConfDF['sentence_nr'] <= id_last_read_sentence]
        promptAlignConfDF_not_read = promptAlignConfDF[promptAlignConfDF['sentence_nr'] > id_last_read_sentence]
        
        read_prompts = " ".join(promptAlignConfDF_read['prompt'])
        read_promptIDs = promptAlignConfDF_read.index
        
        not_read_prompts = " ".join(promptAlignConfDF_not_read['prompt'])
        not_read_promptIDs = promptAlignConfDF_not_read.index
        empty_recWordsDF =  pd.DataFrame([], columns= ['label', 'confidence', 'start', 'end'])

        # The alignment should be done again: read_prompts vs asr_transcript & not_read_prompt vs empty string
        promptAlignConfDF_read, insertionDF_read = alignWithConfidenceScores(asrTranscription, recWordsDF, read_prompts, read_promptIDs)
        promptAlignConfDF_not_read, insertionDF_not_read = alignWithConfidenceScores(' ', empty_recWordsDF, not_read_prompts, not_read_promptIDs)

        newPromptAlignConfDF = pd.concat([promptAlignConfDF_read, promptAlignConfDF_not_read])
        newPromptAlignConfDF['index'] = range(len(newPromptAlignConfDF))
        return newPromptAlignConfDF.loc[:, newPromptAlignConfDF.columns != 'sentence_nr'], insertionDF_read, sentenceStats
    else:
        return promptAlignConfDF.loc[:, promptAlignConfDF.columns != 'sentence_nr'], insertionDF, sentenceStats

# @timeoutable()
def alignWithConfidenceScores(asrTranscription, recWordsDF, prompt, promptIDs):

    # Align prompt and AsrResult transcription
    promptAlignDF = alignOneFile(asrTranscription, prompt)
    # promptAlignDF.to_csv('promptAlignDF.tsv', sep='\t')

    # Add confidence scores with AsrResult
    promptAlignConfDF, insertionDF = addConfidenceScores(promptAlignDF, recWordsDF)
    # promptAlignDF.to_csv('promptAlignConfDF.tsv', sep='\t')

    # Add promptIDs
    promptAlignConfDF['promptID'] = promptIDs
    promptAlignConfDF = promptAlignConfDF.set_index('promptID')

    return promptAlignConfDF, insertionDF

    
def createOutputDirectories(list_of_output_dirs):

    for output_dir in list_of_output_dirs:
        if not os.path.exists(output_dir):
            print("Create output dir:", output_dir)
            os.makedirs(output_dir)


def run(args):

    # Read input variables
    outputDir = args.output_dir
    promptDir = args.prompt_dir
    asrResultDir = args.input_asr_dir


    # Create output directories if they don't exist yet.
    outputDirCsvAlignForward = os.path.join(outputDir, 'csv-align-forward')
    outputDirCsvAlignForIns = os.path.join(outputDir, 'csv-align-forward-ins')
    createOutputDirectories([outputDirCsvAlignForward, outputDirCsvAlignForIns])    
    
    # List all .json files in json_asr_result
    jsonFileList = glob.glob(os.path.join(asrResultDir, '*.json'))
    print('Nr of ASR result files to align: ', len(jsonFileList))

    # Iterate over each audio file, select the corresponding asrResult and prompt, align the two, save csv output in outputDir
    sentenceStatsList = []
    for idx, jsonFile in enumerate(jsonFileList):

        basename = os.path.basename(jsonFile).replace('.json', '')
        task = basename.split('-')[1]

        promptFile = os.path.join(promptDir, task + '.prompt')
        promptIdxFile = os.path.join(promptDir, task + '-wordIDX.csv')
        asrResultFile = os.path.join(asrResultDir, basename + '.json')
        
        # Check if csv alignment file already exists. If not, do the alignment process.
        forwardExists = os.path.isfile(os.path.join(outputDir, 'csv-align-forward/' + basename + '.csv'))
        # if not forwardExists:

        print(basename)

        checkIfFilesExist(asrResultFile, promptFile)

        # Read <task>.prompt file
        prompt = readPromptFile(promptFile)
    
        # Read <task>-wordIDX.csv file
        promptIDs = getPromptIdxs(promptIdxFile)

        # Read .json asrResult file
        asrTranscription, recWordsDF = readAsrResult(asrResultFile)

        # Perform alignment process
        promptAlignConfDF, insertionDF = alignWithConfidenceScores(asrTranscription, recWordsDF, prompt, promptIDs)

        # Correct alignment process in case the whole prompt is not read
        promptAlignConfDF, insertionDF, sentenceStats = correctForNotReadSentences(promptAlignConfDF, insertionDF, asrTranscription, recWordsDF)

        # Save the csv alignment output files
        promptAlignConfDF.to_csv(os.path.join(outputDir, 'csv-align-forward/' + basename + '.csv'))
        insertionDF.to_csv(os.path.join(outputDir, 'csv-align-forward-ins/' + basename + '.csv'))
        
        # Add the sentenceStats to an overview file
        sentenceStatsList.append([basename] + sentenceStats)

        if (idx+1)%10==0:
            print(datetime.now(), ':', idx, 'of', len(jsonFileList), 'json files processed.')

    sentenceStatsDF = pd.DataFrame(sentenceStatsList, columns = ['audioID', 'id_last_read_sentence', 'nr_missing_sentences', 'perc_sentences_read', 'nr_sentences_prompt']).set_index('audioID')
    sentenceStatsDF.to_csv(os.path.join(outputDir, 'sentenceStats.csv'))

    print("Script 01 completed: The prompts of all task files are aligned with the ASR results.")

    print('Nr of files in ', os.path.join(outputDir, 'csv-align-forward:') , len(glob.glob(os.path.join(outputDir, 'csv-align-forward/*.csv'))))
    print('Nr of files in ', os.path.join(outputDir, 'csv-align-forward-ins:') , len(glob.glob(os.path.join(outputDir, 'csv-align-forward-ins/*.csv'))))
    print('sentenceStats.csv created:', os.path.exists(os.path.join(outputDir, 'sentenceStats.csv')))   

def main():
    parser = argparse.ArgumentParser("Message")
    parser.add_argument("--output_dir", type=str, help = "Output directory where csv file with alignment between whisperT output and prompt are saved.")
    parser.add_argument("--prompt_dir", type=str, help = "promptDir")
    parser.add_argument("--input_asr_dir", type=str, help = "Directory with JSON WhisperT AsrResult files corresponding to audio.")

    parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
