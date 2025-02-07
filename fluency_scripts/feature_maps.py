import pandas as pd

# auto name (as in separate feature files), print name of auto variable, corresponding ot variable
v1 = pd.DataFrame([
    # Speech rate
    ['speechrate(nsyll/dur)', 'dejong_sr(nsyll/sec)', 'SpeechRate(nrSyllPerSecond)', 'ot_sr(nsyll/sec)'],
    ['speechRate(WPM)_wht', 'wht_sr(nwords/min)', 'SpeechRate(nrWordsPerMinute)', 'ot_sr(nwords/min)'],
    ['speechRate(WPM)_wht_dis', 'wht-dis_sr(nwords/min)', 'SpeechRate(nrWordsPerMinute)', 'ot_sr(nwords/min)'],
    ['speechRate(WPM)_wht_vad_dis', 'wht-vad-dis_sr(nwords/min)', 'SpeechRate(nrWordsPerMinute)', 'ot_sr(nwords/min)'],
    ['speechRate(WPM)_wht_prompts', 'wht-prompts_sr(nwords/min)', 'SpeechRate(nrWordsPerMinute)', 'ot_sr(nwords/min)'],
    
    # Articulation rate
    ['articulation_rate(nsyll/phonationtime)', 'dejong_ar(nsyll/sec)', 'ArtRate(nrSyllPerSecond)', 'ot_ar(nsyll/sec)'],
    ['articulationRate_wht', 'wht_ar(nwords/min)', 'ArtRate(nrWordsPerMinute)', 'ot_ar(nwords/min)' ], # misschien asr_ar(nwords/min) nog omzetten?
    ['articulationRate_wht_dis', 'wht-dis_ar(nwords/min)', 'ArtRate(nrWordsPerMinute)', 'ot_ar(nwords/min)' ],
    ['articulationRate_wht_vad_dis', 'wht-vad-dis_ar(nwords/min)', 'ArtRate(nrWordsPerMinute)', 'ot_ar(nwords/min)' ],
    ['articulationRate_wht_prompts', 'wht-prompts_ar(nwords/min)', 'ArtRate(nrWordsPerMinute)', 'ot_ar(nwords/min)' ],

    # Mean pause duration (all silent pauses)
    ['MeanUnvoicedSegmentLength', 'egemaps_allpausedur_mean(sec)', 'meanDurAllSilentPauses', 'ot_allpausedur_mean(sec)'],
    ['pauses_dur_mean_wht', 'wht_allpausedur_mean(sec)', 'meanDurAllSilentPauses', 'ot_allpausedur_mean(sec)'],
    ['pauses_dur_mean_wht_dis', 'wht-dis_allpausedur_mean(sec)', 'meanDurAllSilentPauses', 'ot_allpausedur_mean(sec)'],
    ['pauses_dur_mean_wht_vad_dis', 'wht-vad-dis_allpausedur_mean(sec)', 'meanDurAllSilentPauses', 'ot_allpausedur_mean(sec)'],
    ['pauses_dur_mean_wht_prompts', 'wht-prompts_allpausedur_mean(sec)', 'meanDurAllSilentPauses', 'ot_allpausedur_mean(sec)'],

    # Std pause duration (all silent pauses)
    ['StddevUnvoicedSegmentLength', 'egemaps_allpausedur_std(sec)', 'stdDurAllSilentPauses', 'ot_allpausedur_std(sec)'],
    ['pauses_dur_std_wht', 'wht_allpausedur_std(sec)', 'stdDurAllSilentPauses', 'ot_allpausedur_std(sec)'],
    ['pauses_dur_std_wht_dis', 'wht-dis_allpausedur_std(sec)', 'stdDurAllSilentPauses', 'ot_allpausedur_std(sec)'],
    ['pauses_dur_std_wht_vad_dis', 'wht-vad-dis_allpausedur_std(sec)', 'stdDurAllSilentPauses', 'ot_allpausedur_std(sec)'],
    ['pauses_dur_std_wht_prompts', 'wht-prompts_allpausedur_std(sec)', 'stdDurAllSilentPauses', 'ot_allpausedur_std(sec)'],

    # Pause rate: nr of silent pauses per minute
    # ['', '', 'nrOfSilentPausesPerMinute' : 'ot_allpauserate(npause/min)'],

    # Intersentential silent pauses
    # ['inter_mean', '', 'meanDurInterSilentPauses', 'ot_interpausedur_mean(sec)'],
    # ['inter_std', '', 'stdDurInterSilentPauses', 'ot_interpausedur_std(sec)'],

    # Intrasentential silent pauses
    # ['intra_mean', '', 'meanDurationIntraSilentPauses', 'ot_intrapausedur_mean(sec)'],
    # ['intra_std', '', 'stdDurationIntraSilentPauses', 'ot_intrapausedur_std(sec)'],
    # ['', '', 'nrIntraSilentPausesPerMin', 'ot_intrapauserate(npause/min)'],

    # File-level Pitch
    # ['F0semitoneFrom27.5Hz_sma3nz_amean', 'egemaps_pitch_mean(semitones)', ''],
    # ['F0semitoneFrom27.5Hz_sma3nz_stddevNorm', 'egemaps_pitch_std(semitones)', ''],

    # File-level Loudness
    # ['loudness_sma3_amean', 'egemaps_loudness_mean(dB)', ''],
    # ['loudness_sma3_stddevNorm', 'egemaps_loudness_std(dB)', '']

    # Accuracy features (wcpm, perc corr)
    # ['sub_perc', '', '', ''],
    # ['del_perc', '', '', ''],
    # ['ins_perc', '', '', ''],
    # ['cor_perc', '', '', ''],
    # ['cor_prompt_perc', '', '', ''],

], columns = ['auto_orig', 'auto_print', 'ot_orig', 'ot_print'])

v1_autoNameMap = v1[['auto_orig', 'auto_print']].set_index('auto_orig').to_dict()['auto_print']
v1_otNameMap = v1[['ot_orig', 'ot_print']].set_index('ot_orig').to_dict()['ot_print']

v1_autoOtMap = v1[['auto_print', 'ot_print']].set_index('auto_print').to_dict()['ot_print']

# v1_nameMap = {
#     # Original Name : # Name for Max
#     'speechrate(nsyll/dur)' : 'dejong_speechrate(nsyll/sec)',
#     # 'VoicedSegmentsPerSec' : 'egemaps_speechrate(nsyll/sec)',
#     'speechRate(WPM)' : 'asr_speechrate(nwords/min)', # misschien dit nog omzetten?

#     'articulation_rate(nsyll/phonationtime)' : 'dejong_articulationrate(nsyll/sec)',
#     'articulationRate' : 'asr_articulationrate(nwords/min)', # misschien dit nog omzetten?

#     'MeanUnvoicedSegmentLength' : 'egemaps_allpausedur_mean(sec)',
#     'StddevUnvoicedSegmentLength' : 'egemaps_allpausedur_std(sec)',
#     'pauses_dur_mean' : 'asr_allpausedur_mean(sec)',
#     'pauses_dur_std' : 'asr_allpausedur_std(sec)',

#     'F0semitoneFrom27.5Hz_sma3nz_amean' : 'egemaps_pitch_mean(semitones)',
#     'F0semitoneFrom27.5Hz_sma3nz_stddevNorm' : 'egemaps_pitch_std(semitones)',

#     'loudness_sma3_amean' : 'egemaps_loudness_mean(dB)',
#     'loudness_sma3_stddevNorm' : 'egemaps_loudness_std(dB)',
# }

# => pause rate is missing, can we add this? Are the number of pauses counted in one of the methods?

# v1_autoOtMap = {
#     # Automatic feature : # Reference (ort trans) feature
#     'dejong_speechrate(nsyll/sec)' : 'ot_speechrate(nsyll/sec)',
#     'asr_speechrate(nwords/min)' : 'ot_speechrate(nwords/min)', 

#     'dejong_articulationrate(nsyll/sec)' : 'ot_articulationrate(nsyll/sec)',
#     'asr_articulationrate(nwords/min)' : 'ot_articulationrate(nwords/min)', 

#     'egemaps_allpausedur_mean(sec)' : 'ot_allpausedur_mean(sec)',
#     'asr_allpausedur_mean(sec)' : 'ot_allpausedur_mean(sec)',
#     'egemaps_allpausedur_std(sec)' : 'ot_allpausedur_std(sec)',
#     'asr_allpausedur_std(sec)' : 'ot_allpausedur_std(sec)',
#     #'' : 'ot_allpauserate(npause/min)',
#     }