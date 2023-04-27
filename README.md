# Build a Wake Word Detection model for Voice Assistant
This is a Wake Word Detection model for Voice Assistant, which can detect/recognize user's voice (after trained), wake and response to user whenever a wake word from user's speech is detected.

## Running on native machine
### Dependencies
* python3
* portaudio (for recording with pyaudio to work)
* using a linux/mac machine or wsl2 on windows (because torchaudio may not work on windows)
### pip packages
`pip install -r requirements.txt` 

## Running with Docker
### setup
If you are running with just the cpu
`docker build -f cpu.Dockerfile -t voiceassistant .`

If you are running on a cuda enabled machine 
`docker build -f Dockerfile -t voiceassistant .`

## Wake word detection

### scripts
For more details make sure to visit these files to look at script arguments and description

`wakeword/neuralnet/train.py` is used to train the model

`wakeword/neuralnet/optimize_graph.py` is used to create a production ready graph that can be used in `engine.py`

`wakeword/engine.py` is used to demo the wakeword model

`wakeword/scripts/collect_wakeword_audio.py` - used to collect wakeword and environment data

`wakeword/scripts/split_audio_into_chunks.py` - used to split audio into n second chunks

`wakeword/scripts/split_commonvoice.py` - if you download the common voice dataset, use this script to split it into n second chunks

`wakeword/scripts/create_wakeword_jsons.py` - used to create the wakeword json for training

### Steps to train and demo your wakeword model

For more details make sure to visit these files to look at script arguments and description

1. collect data
    1. environment and wakeword data can be collected using `python collect_wakeword_audio.py`
       ```
       cd VoiceAssistant/wakeword/scripts
       mkdir data
       cd data
       mkdir 0 1 wakewords environment
       
       python collect_wakeword_audio.py --sample_rate 8000 --seconds 2 --interactive --interactive_save_path ./data/wakewords

       python collect_wakeword_audio.py --sample_rate 8000 --save_path ./data/environment/env.wav

       ```
    2. to avoid the imbalanced dataset problem, we can duplicate the wakeword clips with 
       ```
       python replicate_audios.py --wakewords_dir data/wakewords/ --copy_destination data/1/ --copy_number 50
       ```
    3. be sure to collect other speech data like random speech or [common voice dataset](https://commonvoice.mozilla.org/en/datasets). split the data into n seconds chunk with `split_audio_into_chunks.py`.
    4. put data into two seperate directory named `0` and `1`. `0` for non wakeword, `1` for wakeword. use `create_wakeword_jsons.py` to create train and test json
    5. create a train and test json in this format...
        ```
        // make each sample is on a seperate line
        {"key": "/path/to/audio/sample.wav, "label": 0}
        {"key": "/path/to/audio/sample.wav, "label": 1}
        ```

2. train model
    1. use `train.py` to train model (--no_cuda for training on just cpu)
        ```
        python train.py --save_checkpoint_path checkpoint/ --train_data_json data_json/train.json --test_data_json data_json/test.json --no_cuda
        ```

    2. after model training us `optimize_graph.py` to create an optimized pytorch model

3. test
    1. test using the `engine.py` script
    
NOTE: the test response will just randomly play audio file in demo_voice/terminator (Terminator's voice), which was downloaded from https://www.soundboard.com/sb/Pappa

*** This is a modified copy from a part of [Michael Nguyen's AI Voice Assistant](https://github.com/LearnedVector/A-Hackers-AI-Voice-Assistant), license included.
## Training result
