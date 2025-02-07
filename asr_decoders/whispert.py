import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

import json
import glob
from datetime import datetime
import whisper_timestamped as whisper
import argparse

import torch
torch.cuda.empty_cache()

import sys
import os
from subprocess import call
os.environ["CUDA_VISIBLE_DEVICES"]='1'
print('_____Python, Pytorch, Cuda info____')
print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDA RUNTIME API VERSION')
#os.system('nvcc --version')
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('_____nvidia-smi GPU details____')
call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
print('_____Device assignments____')
print('Number CUDA Devices:', torch.cuda.device_count())
print ('Current cuda device: ', torch.cuda.current_device(), ' **May not correspond to nvidia-smi ID above, check visibility parameter')
print("Device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))

###
# $ nohup time python3 whispert.py &
####y

def run(args):

    audioDir = args.audioDir
    audioExtension = args.audioExtension
    spkTaskSep = args.spkTaskSep # "-" or "_"
    promptsDir = args.promptsDir
    asrSettings = args.asrSettings
    outputDir = args.asrResultDir

    audioFileList = glob.glob(os.path.join(audioDir, '*'+ audioExtension))
                              
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    startTime = datetime.now()

    for audioFile in audioFileList:

        output_json_file = os.path.join(outputDir, os.path.basename(audioFile).replace(audioExtension, '.json'))
        print(output_json_file)

        if not os.path.exists(output_json_file):
            print('Create: ', os.path.basename(audioFile).replace(audioExtension, '.json'))
            
            # Read corresponding prompt
            taskID = os.path.basename(audioFile).split(spkTaskSep)[1].replace(audioExtension, '')
            taskFile = os.path.join(promptsDir, taskID + '.prompt')

            with open(taskFile, 'r') as f:
                taskPrompt = f.readlines()[0]

            audio = whisper.load_audio(audioFile)

            model = whisper.load_model(name="large-v2", download_root="/vol/tensusers5/wharmsen/whisper/.cache")

            # Parse ASR settings
            det_dis=False
            if 'dis' in asrSettings:
                det_dis=True
                
            if 'prompt' not in asrSettings:
                taskPrompt = None
            
            vad_boolean = False
            if 'vad' in asrSettings:
                vad_boolean = True

            # parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
            try:
                result = whisper.transcribe(model, audio, language="nl", detect_disfluencies=det_dis, vad=vad_boolean, initial_prompt=taskPrompt)
                with open(output_json_file, 'w') as f:
                    f.write(json.dumps(result, indent = 2, ensure_ascii = False))
            except:
                print('Error for', os.path.basename(audioFile))

    endTime = datetime.now()

    print("Done: processed", len(audioFileList), "audio files from " ,startTime,  "till", endTime)

def main():
    parser = argparse.ArgumentParser("Message")
    parser.add_argument("--asrSettings", type=str, help = "choose from: whispert, whispert_dis, whispert_prompt or whispert_dis_prompt")
    parser.add_argument("--audioDir", type=str, help = "Path to preprocessed audio directory.")
    parser.add_argument("--audioExtension", type=str, help = "Extension of audio files in audio dir (i.e., .wav or .mp3)")
    parser.add_argument("--spkTaskSep", type=str, help = "Speaker task separator. The audio files are named according to the following convention: <spk>-<task>-<attempt>.wav. The - is the spkTaskSep.")
    parser.add_argument("--promptsDir", type=str, help = "Path to prompts directory. Contains for each task a file <task>.prompt")
    parser.add_argument("--asrResultDir", type=str, help = "Path to json-asr-results directory. This is the output directory.")

    parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
