"""
Not all features produced by GeMAPS and eGeMAPS are useful for oral reading fluency.
In this script, we select the relevant features.
"""

import opensmile
import glob
import pandas as pd
import argparse
import os

def renameFile(fileName):
    spk = fileName.split('-')[0]
    task = fileName.split('-')[1]
    return spk + '-' + task

def run(args):
    
    egemaps_file = args.eGeMAPSFile
    print("02_eGeMAPS_feat_sel - input tsv: ", egemaps_file)

    df = pd.read_csv(egemaps_file, index_col=0, sep='\t')

    """
    Pitch related features (all computed on voiced regions, i.e. F0 != 0):
    F0semitoneFrom27.5Hz_sma3nz_amean	
    F0semitoneFrom27.5Hz_sma3nz_stddevNorm	        Variation in F0
    F0semitoneFrom27.5Hz_sma3nz_percentile20.0 
    F0semitoneFrom27.5Hz_sma3nz_percentile50.0	
    F0semitoneFrom27.5Hz_sma3nz_percentile80.0	
    F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2        Variation in F0 (the range of 20-th to 80-th percentile)
    F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope	
    F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope	
    F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope 
    F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope

    Loudness related features (computed on voiced and unvoiced regions)
    loudness_sma3_amean                             Mean loudness (the range of 20-th to 80-th percentile)
    loudness_sma3_amean	loudness_sma3_stddevNorm	Variation in loudness
    loudness_sma3_percentile20.0
    loudness_sma3_percentile50.0	
    loudness_sma3_percentile80.0	
    loudness_sma3_pctlrange0-2	                    Variation in loudness (the range of 20-th to 80-th percentile)
    loudness_sma3_meanRisingSlope	
    loudness_sma3_stddevRisingSlope	
    loudness_sma3_meanFallingSlope	
    loudness_sma3_stddevFallingSlope

    VoicedSegmentsPerSec	      Pseudo Syllable Rate   
    MeanUnvoicedSegmentLength	  Approximating pauses: mean pause duration
    StddevUnvoicedSegmentLength   Approximating pauses: variation in pause duration
    """
    

    pitch_features = ['F0semitoneFrom27.5Hz_sma3nz_amean', 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm', 'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2']
    loudness_features = ['loudness_sma3_amean', 'loudness_sma3_stddevNorm', 'loudness_sma3_pctlrange0-2']
    rate_and_pause_features = ['VoicedSegmentsPerSec', 'MeanUnvoicedSegmentLength', 'StddevUnvoicedSegmentLength']
    other_features = []

    feature_selection = pitch_features + loudness_features + rate_and_pause_features + other_features

    df_new = df.loc[:, feature_selection]
    df_new['VoicedSegmentsPerMin'] = df_new['VoicedSegmentsPerSec']*60

    df_new['audioID'] = [renameFile(os.path.basename(x)) for x in df_new.index]
    df_new = df_new.set_index('audioID')

    output_file = egemaps_file.replace('.tsv', '_selected.tsv')
    df_new.to_csv(output_file, sep='\t')
    print("02_eGeMAPS_feat_sel - output tsv: ", output_file)

def main():
    parser = argparse.ArgumentParser("Message")
    parser.add_argument("--eGeMAPSFile", type=str, help = "Path to tsv file with eGeMAPS features and functionals.")

    parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()