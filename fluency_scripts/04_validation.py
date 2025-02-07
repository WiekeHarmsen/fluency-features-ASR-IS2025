import argparse
import os
import pandas as pd
import pandas as pd 
import os
from sklearn.metrics import root_mean_squared_error, r2_score
from scipy import stats

import feature_maps


def getSyllableSpeechRate(row):
    nsyll = row['syllable_count']
    dur = row['totalDuration']
    return nsyll/dur

def getSyllableArtRate(row):
    nsyll = row['syllable_count']
    phontime = row['phonationTime']
    return nsyll/phontime

def readAndCombineAutoFeatFiles(autoBasePath, nameMap):

    # Read all separate feature files
    combinedDFList = []

    deJongFeaturesFile = os.path.join(autoBasePath,'de_jong_syll_nucl.tsv')
    deJongDF = pd.read_csv(deJongFeaturesFile, sep='\t', index_col=0).sort_index()
    combinedDFList.append(deJongDF)

    eGeMAPSFeatureFile =  os.path.join(autoBasePath,'eGeMAPSv02_Functionals_88feat.tsv')
    egemapsDF = pd.read_csv(eGeMAPSFeatureFile, sep='\t')
    egemapsDF['file'] = egemapsDF['file'].apply(lambda x: os.path.basename(x).replace('.wav', ''))
    egemapsDF['audioID'] = egemapsDF['file'].apply(lambda x: x.split('-2')[0])
    egemapsDF = egemapsDF.set_index('audioID')
    combinedDFList.append(egemapsDF)

    asrSettingsList = ['whispert', 'whispert_dis', 'whispert_vad_dis', 'whispert_prompts']

    for asrSettings in asrSettingsList:
        asrFeatureFile = os.path.join(autoBasePath, asrSettings + '-asr-features.tsv')
        asrDF = pd.read_csv(asrFeatureFile, sep='\t', index_col=0)
        asrDF.index = [x.split('-2')[0] for x in asrDF.index]
        asrDF.columns = [x + '_' + asrSettings.replace('whispert', 'wht') for x in asrDF.columns]
        combinedDFList.append(asrDF)

    # Combine all dataframe into one dataframe
    combinedDF = pd.concat(combinedDFList, axis=1)

    # Select features from the specified feature map
    combinedDF_renamed = combinedDF.loc[:, nameMap.keys()].rename(columns=nameMap)

    return combinedDF_renamed

def readAndCombineOtFeatFiles(otBasePath, otNameMap):
    pacePhrasePath = os.path.join(otBasePath,'pacePhrasingDF.tsv')
    pacePhraseDF = pd.read_csv(pacePhrasePath, sep='\t', index_col=0)

    syllablePath = os.path.join(otBasePath,'syllableCountDF.tsv') # Create this using: /vol/tensusers5/wharmsen/dartastla/syllabificator/use_syllabificator.ipynb
    syllableDF = pd.read_csv(syllablePath, sep='\t', index_col=0)

    pacePhraseDF = pd.concat([pacePhraseDF, syllableDF], axis=1)
    pacePhraseDF['SpeechRate(nrSyllPerSecond)'] = pacePhraseDF.apply(getSyllableSpeechRate, axis=1)
    pacePhraseDF['ArtRate(nrSyllPerSecond)'] = pacePhraseDF.apply(getSyllableArtRate, axis=1)

    # Select features manx
    otFeaturesDF = pacePhraseDF.loc[:, otNameMap.keys()].rename(columns=otNameMap)

    # Rename index column (only keep speaker and task)
    otFeaturesDF.index = [x.split('-20')[0] for x in otFeaturesDF.index]
    
    return otFeaturesDF

def comparisonOtAutoFeatures(autoOtDF, autoOtMap):

    outputList = []
    for auto_var, ot_var in autoOtMap.items():

        rmse = root_mean_squared_error(autoOtDF[ot_var], autoOtDF[auto_var])
        corr = stats.pearsonr(autoOtDF[ot_var], autoOtDF[auto_var])[0]
        r2 = r2_score(autoOtDF[ot_var], autoOtDF[auto_var])
        mean_auto_var = autoOtDF[auto_var].mean()
        std_auto_var = autoOtDF[auto_var].std()
        mean_ot_var = autoOtDF[ot_var].mean()
        std_ot_var = autoOtDF[ot_var].std()
        outputList.append([auto_var, ot_var] + [round(x, 4) for x in [mean_auto_var, std_auto_var, mean_ot_var, std_ot_var, rmse, corr, r2]])

    outputDF = pd.DataFrame(outputList, columns = ['auto_var', 'ot_var', 'mean_auto', 'std_auto', 'mean_ot', 'std_ot', 'rmse', 'corr','r2'])
    
    return outputDF

def DF2Latex(outputDF, column_list, outputFile, txt):
    latex_table = outputDF.to_latex(index=False,
                        columns = column_list,
                        formatters={"name": str.upper},
                        float_format="{:.2f}".format,
                        caption=txt)
    
    with open(outputFile, 'w') as f:
        f.write(latex_table.replace('_', '-'))
    

def run(args):
    print('Start 04_Validation.p')

    # Set base path
    # basePath = '/vol/tensusers2/wharmsen/JASMIN-fluency-features/comp-q-read_nl_age7-11_nat'
    basePath = args.basePath # INPUT 1
    autoBasePath = os.path.join(basePath, '05_automatic_fluency_features')
    otBasePath = os.path.join(basePath, '06_manual_fluency_features')
    validationBasePath = args.outputDir # os.path.join(basePath, '07_validation_exp')

    # Set feature map
    featureMapKey = args.featureMapKey # INPUT 2 'maxFeatureMap_v1'
    if featureMapKey == 'v1':
        autoNameMap = feature_maps.v1_autoNameMap
        otNameMap = feature_maps.v1_otNameMap
        autoOtMap = feature_maps.v1_autoOtMap
    else:
        autoNameMap = None
        otNameMap = None
        autoOtMap = None
    assert autoNameMap != None, 'select a correct feature map, choose from feature_maps.py'

    # Read and create output dir
    valOutputDir = os.path.join(validationBasePath, featureMapKey)

    if not os.path.exists(valOutputDir):
        os.makedirs(valOutputDir)


    ####################################################
    ### PART 1: Read and combine auto feature files  ###
    ####################################################

    combinedAutoDF_renamed = readAndCombineAutoFeatFiles(autoBasePath, autoNameMap)

    # Save the selected features in a .tsv file
    combinedAutoDF_renamed.to_csv(os.path.join(valOutputDir, '01_auto_features.tsv'), sep='\t')

    ####################################################
    ### PART 2: Read and combine ot feature files    ###
    ####################################################

    selectedOtDF_renamed = readAndCombineOtFeatFiles(otBasePath, otNameMap)
    selectedOtDF_renamed.to_csv(os.path.join(valOutputDir, '02_ot_features.tsv'), sep='\t')
    
    ####################################################
    ### PART 3: Comparison automatic and ot features ###
    ####################################################

    # Combine auto and OT features
    autoOtDF = pd.concat([combinedAutoDF_renamed, selectedOtDF_renamed], axis=1)
    print('Length autoOtDF before dropping NA', len(autoOtDF))
    autoOtDF = autoOtDF.dropna()
    print('Length autoOtDF after dropping NA', len(autoOtDF))
    autoOtDF.to_csv(os.path.join(valOutputDir, '03_auto_ot_features.tsv'), sep='\t')

    metricsDF = comparisonOtAutoFeatures(autoOtDF, autoOtMap)

    evalOutputFile = os.path.join(valOutputDir, '04_eval_auto_features.tsv')
    metricsDF.to_csv(evalOutputFile, sep='\t')

    DF2Latex(metricsDF, ['auto_var', 'rmse', 'corr', 'r2'], os.path.join(valOutputDir, '04_eval_auto_features.txt'), 'validation')

    DF2Latex(metricsDF, ['auto_var', 'ot_var', 'mean_auto', 'mean_ot', 'std_auto', 'std_ot'], os.path.join(valOutputDir, '05_descrstats_auto_ot.txt'), 'descriptive statistics')

    print('End 04_Validation.py, see results: ', valOutputDir)

def main():
    parser = argparse.ArgumentParser("Message")
    parser.add_argument("--basePath", type=str, help = "")
    parser.add_argument("--featureMapKey", type=str, help = "")
    parser.add_argument("--outputDir", type=str, help = "")

    parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()