#!/bin/bash

######################
### INPUT settings ###
######################

# datasetDir=/vol/tensusers2/wharmsen/JASMIN-fluency-features/comp-q-read_nl_age7-11_nat
# datasetDir=/vol/tensusers2/wharmsen/JASMIN-fluency-features/comp-q-read_vl_age7-11_nat
datasetDir=/vol/tensusers2/wharmsen/SERDA-fluency-features/comp1-new

promptsDir=$datasetDir/01_prompts
audioDir=$datasetDir/02_audio
audioExtension=.wav
spkTaskSep=-
asrDir=$datasetDir/04_asr
autoFeatDir=$datasetDir/05_automatic_fluency_features
manualFeatDir=$datasetDir/06_manual_fluency_features

################################
### Compute fluency features ###
################################

# Three different types of fluency features: 
# 1) from audio (a)
# 2) from audio + word segmentation (aw)
# 3) from audio + word segmentation + prompt segmentation (awp)

# Create Fluency dir
# Create outputDir for fluency features if it does not already exist.
if ! test -d $asrDir; then
    mkdir $asrDir;
fi

if ! test -d $autoFeatDir; then
    mkdir $autoFeatDir;
fi

if ! test -d $manualFeatDir; then
    mkdir $manualFeatDir;
fi


######################################################################
####   FLUENCY STEP 1: Compute features directly from audio (a)   ####
######################################################################

### Praat script de Jong et al.: One script to run the Praat script, one to convert the feature txt file to a csv file.

current_datetime=$(date +"%Y-%m-%d_%H:%M:%S.%3N")
echo "Current date and time: $current_datetime"
# output_txt=$autoFeatDir/de_jong_syll_nucl_$current_datetime.txt
# python3 ./fluency_scripts/01_de_jong_syllable_nuclei_v3.py --audioDir $audioDir --audioExtension '.wav' --fluencyDir $autoFeatDir > $output_txt
# python3 ./fluency_scripts/01_de_jong_syllable_nuclei_postprocess.py --fluencyFeatureTxt $output_txt --fluencyFeatureTsv $autoFeatDir/de_jong_syll_nucl.tsv


### eGeMAPS features (using Open Smile)

featureSet=eGeMAPSv02 # GeMAPSv01b, eGeMAPSv02, ComParE_2016 (GeMAPSv01b = 62 features; eGeMAPSv02 = 88 features, GeMAPS is a subset of eGeMAPS. Always use eGeMAPS.)
featureLevel=Functionals # LowLevelDescriptors, LowLevelDescriptors_Deltas, Functionals (Functionals are file-level metrics)
# python3 ./fluency_scripts/02_eGeMAPS.py --featureSet $featureSet --featureLevel $featureLevel --audioDir $audioDir --audioExtension '.wav' --fluencyDir $autoFeatDir
# python3 ./fluency_scripts/02_eGeMAPS_feature_selection.py --eGeMAPSFile $autoFeatDir/eGeMAPSv02_Functionals_88feat.tsv



########################################################################################
###   FLUENCY STEP 2: Compute features directly from audio + word segmentation (aw)  ###
########################################################################################

# for asrSettings in whispert whispert_dis whispert_vad_dis whispert_prompts
# do
#     echo "Step 2: Processing $asrSettings"
    
#     # Decode audio using ASR
#     # python3 ./asr_decoders/whispert.py --asrSettings $asrSettings --audioDir $audioDir --audioExtension $audioExtension --spkTaskSep $spkTaskSep --promptsDir $promptsDir --asrResultDir $asrDir/$asrSettings/json-asr-results

#     # Compute features directly from .json ASR results and create TextGrids
#     python3 ./fluency_scripts/03_asr-results2textgrids.py --jsonAsrResultsDir $asrDir/$asrSettings/json-asr-results --audioDir $audioDir
#     python3 ./fluency_scripts/03_asr-results2features.py --jsonAsrResultsDir $asrDir/$asrSettings/json-asr-results --outputFile $autoFeatDir/$asrSettings/asr-features.tsv
# done



################################################################################################################################
#######      FLUENCY STEP 3: Compute features directly from audio + word segmentation + prompt segmentation (awp)         ######
################################################################################################################################

# for asrSettings in whispert #whispert_dis whispert_vad_dis whispert_prompts
# do
#     echo "STEP 3: Processing $asrSettings"

#     # Align ASR result with prompt
#     python3 ./asr-prompt-aligners/stories-align-prompt-whispert-confStartEnd.py --input_asr_dir $asrDir/$asrSettings/json-asr-results --prompt_dir $promptsDir --output_dir $asrDir/$asrSettings

#     # Compute reading accuracy-related features
#     python3 ./fluency_scripts/05_accuracy_scores.py --asrDir $asrDir --asrSettings $asrSettings --outputDir $autoFeatDir/$asrSettings

#     # Compute intrasentential and intersentential pause rate, duration and std
#     python3 ./fluency_scripts/06_inter-intra-pauses.py --asrDir $asrDir --asrSettings $asrSettings --outputDir $autoFeatDir/$asrSettings
# done


##################################################################################
####   Step 4: Compute the same features from the orthographic transcriptions  ###
##################################################################################

# Compute features directly from .json ASR results and create TextGrids

echo "STEP 4: Computing manual fluency features"

asrSettings=json-fluency-features
otFluencyDir=$manualFeatDir/$asrSettings
if ! test -d $otFluencyDir; then
    mkdir $otFluencyDir;
fi

python3 ./fluency_scripts/03_asr-results2textgrids.py --jsonAsrResultsDir $manualFeatDir/json-orth-trans --audioDir $audioDir
python3 ./fluency_scripts/03_asr-results2features.py --jsonAsrResultsDir $manualFeatDir/json-orth-trans --outputFile $manualFeatDir/$asrSettings/asr-features.tsv

#Align ASR result with prompt
python3 ./asr_prompt_aligners/stories-align-prompt-whispert-confStartEnd.py --input_asr_dir $manualFeatDir/json-orth-trans --prompt_dir $promptsDir --output_dir $manualFeatDir/$asrSettings

# Compute reading accuracy-related features
python3 ./fluency_scripts/05_accuracy_scores.py --asrDir $manualFeatDir --asrSettings $asrSettings --outputDir $manualFeatDir/$asrSettings

# Compute intrasentential and intersentential pause rate, duration and std
python3 ./fluency_scripts/06_inter-intra-pauses.py --asrDir $manualFeatDir --asrSettings $asrSettings --outputDir $manualFeatDir/$asrSettings


# STEP 5: Validate automatic features by comparing them to orthographic transcriptions
# python3 ./fluency_scripts/04_validation.py --basePath $datasetDir --featureMapKey v1 --outputDir $datasetDir/07_validation_exp