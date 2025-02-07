import opensmile
import glob
import pandas as pd
import argparse
import os

def run(args):

    # Initialize Open Smile features
    """
    See: https://audeering.github.io/opensmile-python/
    Paper: https://sail.usc.edu/publications/files/eyben-preprinttaffc-2015.pdf

    Currently, three standard sets are supported. 
    ComParE 2016 is the largest with more than 6k features. 
    The smaller sets GeMAPS and eGeMAPS come in variants v01a, v01b and v02 (only eGeMAPS). 
    We suggest to use the latest version unless backward compatibility with the original papers is desired.
    
    Each feature set can be extracted on two levels:
    1) Low-level descriptors (LDD)
    2) Functionals

    For ComParE 2016 a third level is available:
    3) LLD deltas

    Feature description:
    The minimalistic acoustic parameter set contains the following compact set of 18 Low-level descriptors (LLD).
    These 18 low-level descriptors are sorted over three parameter groups: Frequency related parameters,
    energy/amplitude related parameters, spectral (balance) parameters.

    All LLD are smoothed over time with a symmetric moving average filter 3 frames long (for pitch, jitter,
    and shimmer, the smoothing is only performed within voiced regions, i.e., not smoothing the transitions
    from 0 (unvoiced) to non 0). Arithmetic mean and coefficient of variation (standard deviation normalised
    by the arithmetic mean) are applied as functionals to all 18 LLD, yielding 36 parameters. 

    To loudness and pitch the following 8 functionals are additionally applied: 20-th, 50-th, and 80-th percentile, the range of
    20-th to 80-th percentile, and the mean and standard deviation of the slope of rising/falling signal parts.
    All functionals are applied to voiced regions only (non-zero F0), with the exception of all the functionals
    which are applied to loudness. This gives a total of 52 parameters.

    Also, the arithmetic mean of the Alpha Ratio, the Hammarberg Index, and the spectral slopes
    from 0–500 Hz and 500–1500 Hz over all unvoiced segments are included, totalling 56 parameters.

    In addition, 6 temporal features are included:
    • the rate of loudness peaks, i. e., the number of loudness peaks per second,
    • the mean length and the standard deviation of continuously voiced regions (F0 > 0),
    • the mean length and the standard deviation of unvoiced regions (F0 = 0; approximating pauses),
    • the number of continuous voiced regions per second (pseudo syllable rate).

    In total, 62 parameters are contained in the Geneva Minimalistic Standard Parameter Set.

    eGeMAPS
    Thus, an extension set to the minimalistic set is proposed which contains the following 7 LLD in addition to the 18
    LLD in the minimalistic set. As functionals, the arithmetic mean and the coefficient of variation are applied to all of these 7
    additional LLD to all segments. (voiced and unvoiced together), except for the formant bandwidths to which the functionals are 
    applied only in voiced regions. This adds 14 extra descriptors.

    Additionally, the arithmetic mean of the spectral flux in unvoiced regions only, the arithmetic mean and coefficient of variation
    of the spectral flux and MFCC 1–4 in voiced regions only is included. This results in another 11 descriptors. Additionally the 
    equivalent sound level is included. This results in 26 extra parameters. In total, when combined with the Minimalistic Set, the extended
    Geneva Minimalistic Acoustic Parameter Set (eGeMAPS) contains 88 parameters.
    """

    featureSetMap = {
        'GeMAPSv01b' : opensmile.FeatureSet.GeMAPSv01b,
        'eGeMAPSv02' : opensmile.FeatureSet.eGeMAPSv02,
        'ComParE_2016' : opensmile.FeatureSet.ComParE_2016,
    }

    featureLevelMap = {
        'LowLevelDescriptors' : opensmile.FeatureLevel.LowLevelDescriptors,
        'LowLevelDescriptors_Deltas' : opensmile.FeatureLevel.LowLevelDescriptors_Deltas,
        'Functionals' : opensmile.FeatureLevel.Functionals,
    }

    featureSet = args.featureSet 
    featureSetKey = featureSetMap[featureSet]
    featureLevel = args.featureLevel
    featureLevelKey = featureLevelMap[featureLevel]

    smile = opensmile.Smile(
        feature_set=featureSetKey, # Choose:  GeMAPSv01b, eGeMAPSv02, ComParE_2016
        feature_level=featureLevelKey, # Choose: LowLevelDescriptors, LowLevelDescriptors_Deltas, Functionals
    )

    audioDir = args.audioDir
    audioExtension = args.audioExtension
    outputDir = args.fluencyDir

    audioFileList = glob.glob(os.path.join(audioDir, '*' + audioExtension))

    assert len(audioFileList) > 0, "In this audioDir are no " + audioExtension + " files."

    # Extract OpenSmile features for all 
    outputDF = pd.DataFrame()
    for audioFile in audioFileList:

        if(len(outputDF.columns )==0):
            outputDF = smile.process_file(audioFile)
        else:
            y = smile.process_file(audioFile)
            outputDF = pd.concat([outputDF,y])

    # outputFile = os.path.join(outputDir, str(featureSet) + '_' + str(featureLevel) + '_' + str(outputDF.shape[0]) + 'row_' + str(outputDF.shape[1]) + 'feat.tsv')
    outputFile = os.path.join(outputDir, str(featureSet) + '_' + str(featureLevel) + '_' + str(outputDF.shape[1]) + 'feat.tsv')
    outputDF.to_csv(outputFile, sep='\t')

    print("02_eGeMAPS - output tsv: ", outputFile)

def main():
    parser = argparse.ArgumentParser("Message")
    parser.add_argument("--featureSet", type=str, help = "String representing featureSet, choose from: GeMAPSv01b, eGeMAPSv02, ComParE_2016")
    parser.add_argument("--featureLevel", type=str, help = "String representing featureLevel, choose from: LowLevelDescriptors, LowLevelDescriptors_Deltas, Functionals")
    parser.add_argument("--audioDir", type=str, help = "Path to audioDir directory.")
    parser.add_argument("--audioExtension", type=str, help = "Audio extension, e.g., .wav or .mp3")
    parser.add_argument("--fluencyDir", type=str, help = "Path to dir where the output file should be saved.")

    parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
