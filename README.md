# StreamMUSE 🎹⏳
This final goal of this task is to generate accompniament given melody by user input in a real-time streaming fashion. We use 'mel' as short for melody and 'acc' as short for accompaniment. 

Below is a simple diagram showing the (1) model architecture and (2) how the encoder captures information from midi. 
<!-- ![My Diagram](images/very_simple_architecture.png) -->
<img src="images/very_simple_architecture.png" alt="My graph" width="400" />
<img src="images/attn.png" alt="My graph" width="400" />


The model uses Roformer as its encoders and decoder. Check its documentation [here](https://huggingface.co/docs/transformers/en/model_doc/roformer). 


## Environment 

Install required packages before you start: 

    pip install -r requirements.txt

We adapted the transformers package for it to take care of the special positional encoding:
    
    cd transformers
    pip install -e .

## Data Preparation

Download POP909 dataset:

    git clone https://github.com/music-x-lab/POP909-Dataset.git

+ you can use Garageband or other online DAW, such as [soundtrap](https://www.soundtrap.com/musicmakers), to visualize MIDI files. 

Run `python extract_mid.py` to extract accompaniment and melody tracks in midi files. 

+ ![TODO](https://img.shields.io/badge/TODO-important-red): the script is implemented for accompaniment, adapt it to separate melody as well. 

Make sure your MIDI dataset follow the below data structure and make sure the same song shares same name in both folders:

- **/name_of_dataset**
    - **/mel**
        - 001.mid
        - 002.mid
        - ...
    - **/acc**
        - 001.mid
        - 002.mid
        - ...

To preprocess the midi files into tensor (input to the model), run
    
    python preprocess_large_midi_dataset.py --folders /path/to/folder1 /path/to/folder2 --name name_of_dataset --polyphony num_of_notes


+ `--folders` let you pass folder(s) together and combine them to one tensor. 
+ `--name` just pass a name.
+ `--polyphony` let you define how many notes are allowed to play simutaneously on one timestep. 

The output tensor will be stored under `/data`, find it there.

## Training 

Train model with:

    python m2a_transformer_w_chord.py --batch_size 10 --model_size small --path_to_dataset data/909correct_acc_cp4.pt --wandb
    python m2a_transformer.py --batch_size 10 --model_size small --path_to_dataset data/909correct_acc_cp4.pt --wandb

+ `--model_size` can be either 'large' or 'small'.
+ `--path_to_dataset` passes tensor of the accompaniment (the model will automatically load the corresponding melody tensor at `xxx_mel.pt`).
+ `--wandb` switch logging from Tensorboard(default) to wandb, configure your wandb inside the script.
+ `--model_name` let you save the model with customized name, but make sure you include 'small' or 'large' in the model name as the indicator of its size.


## Inference

Download checkpoint from [here](https://huggingface.co/Jianshu001/music/blob/main/cp_transformer_909%2Bac%2B1k7_trackemb_interleavepos_v0.2_large_batch_40_schedule.epoch%3D00.val_loss%3D0.90296.ckpt) and place it under `/ckpt`
(for faster download, use wget command)

For inference, place all the melody and accompaniment midi under the `/input` folder:

- **input**
    - **/mel**
        - 001.mid
        - ...
    - **/acc**
        - 001.mid
        - ...

Run inferencen with:

    python m2a_transformer_inference.py --model_path /home/coder/laopo/StreamMUSE/ckpt/m2a_transformer_chord_small_batch_20_schedule/m2a_transformer_chord_small_batch_20_schedule.epoch=00.val_loss=0.92533.ckpt --prompt_len 75 --n_samples 2 --temperature 1.0

    python m2a_transformer_inference_w_chord.py --model_path /home/coder/laopo/StreamMUSE/ckpt/m2a_transformer_v0.2_chord_small_batch_20_schedule/m2a_transformer_v0.2_chord_small_batch_20_schedule.epoch=00.val_loss=0.73335.ckpt --prompt_len 75 --n_samples 2 --temperature 1.0

+ `--prompt_len` let you define the prompt_len.
+ `--n_samples` let you generate multiple output on one input.
+ `--temperature` let you define the temperature.

The result will be saved in `\temp.`
