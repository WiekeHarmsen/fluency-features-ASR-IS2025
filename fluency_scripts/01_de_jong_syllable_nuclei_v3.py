import parselmouth
from parselmouth.praat import call, run_file
import argparse
import os
import shutil
import glob

def run(args):

    audioDir = args.audioDir
    audioExtension = args.audioExtension
    fluencyDir = args.fluencyDir

    audioFilesRegExp = os.path.join(audioDir, '*' + audioExtension)
    assert len(glob.glob(audioFilesRegExp)) > 0, "audioDir doesn't contain " + audioExtension + " audio files"

    audioFileList = [os.path.basename(x) for x in glob.glob(audioFilesRegExp)]

    for audioFile in audioFileList:

        # Write path to audio file with following structure: './comp-q-read_nl_age7-11_nat/audio_first_story/*fn000051.wav'
        audioPath = os.path.join(audioDir, '*' + audioFile)

        try:
            run_file('./fluency_scripts/SyllableNucleiv3.praat', audioPath, 'None', -25, 2, 0.3, True, 'Dutch', 1, 'Praat Info window', 'OverWriteData', True)
        except Exception as e:
            print('\n')
            pass

    # Running this script also automatically creates .TextGrid files in the audioDir
    # With this piece of code, we move these .TextGrid files to the outputDir
    outputDir = os.path.join(fluencyDir, 'de_jong_textgrids')

    # If outputDir doesn't exist, create it and move the TextGrid files to it from the audioDir
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    
        for tg_file in glob.glob(os.path.join(audioDir, '*.TextGrid')):

            if not os.path.exists(os.path.join(outputDir, tg_file)):
                shutil.move(tg_file, outputDir)
    
    # Remove remaining TextGrid files in audioDir
    for tg_file in glob.glob(os.path.join(audioDir, '*.TextGrid')):
        os.remove(os.path.join(audioDir, tg_file))

def main():
    parser = argparse.ArgumentParser("Message")
    parser.add_argument("--audioDir", type=str, help = "Path to audioDir directory.")
    parser.add_argument("--audioExtension", type=str, help = "Audio extension, e.g., .wav or .mp3")
    parser.add_argument("--fluencyDir", type=str, help = "Path to dir where de Jong's output TextGrids should be saved.")

    parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()