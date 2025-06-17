import numpy as np

from m2a_transformer import RoFormerSymbolicTransformer, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
# from preprocess_large_midi_dataset_private import preprocess_midi, DURATION_TEMPLATES
from preprocess_large_midi_dataset import preprocess_midi, DURATION_TEMPLATES

from settings import RWC_DATASET_PATH
import torch
import pretty_midi
import os
import argparse
import json
def decode_output(outputs, save_path, tempo=120.0, prompt=True, single=False):
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    time_step_length = 60.0 / tempo / 4
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    for output in outputs:
        instrument_map = {}
        for time_step, data in enumerate(output):
            content = data.squeeze(0)
            time_step = time_step if single else time_step//2
            start_time = time_step * time_step_length
            for i in range(0, len(content), 2):
                program = int(content[i].item())
                if program ==EOS_TOKEN:
                    break
                if i + 1 >= len(content):
                    print('Incomplete note @', time_step, i)
                    break
                pitch_duration = int(content[i+1].item())
                pitch_duration = pitch_duration - 2
                pitch = pitch_duration % 128
                duration = pitch_duration // 128
                if program != 0 and program != 1:
                    print('Invalid program:', program, '@', time_step, i)
                    break
                if pitch < 0 or pitch >= 128:
                    print('Invalid pitch:', pitch, '@', time_step, i)
                    break
                if duration < 0 or duration >= len(DURATION_TEMPLATES):
                    print('Invalid duration:', duration, '@', time_step, i)
                    break
                end_time = DURATION_TEMPLATES[duration] * time_step_length + start_time
                if program not in instrument_map:
                    if program == 0:
                        inst = pretty_midi.Instrument(program=24, name='Guitar')
                    else:  # program == 1
                        inst = pretty_midi.Instrument(program=0, name='Piano')
                    instrument_map[program] = inst
                    midi.instruments.append(inst)

                inst = instrument_map[program]
                inst.notes.append(
                    pretty_midi.Note(
                        velocity=100,
                        pitch=pitch,
                        start=start_time,
                        end=end_time
                    )
                )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    midi.write(save_path)


def decompress(model, byte_arr_mel, byte_arr_acc):
    x = torch.tensor(byte_arr_mel).unsqueeze(0)
    x = x.cuda()
    y = torch.tensor(byte_arr_acc).unsqueeze(0)
    y = y.cuda()
    return model.preprocess(x, pitch_shift=torch.zeros(1, dtype=torch.int8).cuda(), y=y)

def continuation(model, midi_path, prompt_length=100, generation_length=384, temperature=1.0, n_samples=1, gt_mel=True):
    if os.path.isfile(midi_path):
        pass
    else:
        print(f"Error: {midi_path} is not a valid file.")
    byte_arr_mel = preprocess_midi(midi_path, 4)
    byte_arr_acc = preprocess_midi(midi_path.replace('mel', 'acc'), 4)

    if byte_arr_mel is None:
        print(f"Error: preprocess_midi returned None for mel file: {midi_path}")
        return  # Skip this MIDI file
    if byte_arr_acc is None:
        print(f"Error: preprocess_midi returned None for acc file: {midi_path.replace('mel', 'acc')}")
        return  # Skip this MIDI file

    x_mel, x_acc = decompress(model, byte_arr_mel[0], byte_arr_acc[0])
    
    if prompt_length==0:
        B, S, L = x_mel.shape
        valid = (x_mel != EOS_TOKEN) & (x_mel != PAD_TOKEN)  # [B, S, L]，True 表示该 token 有效
        frame_has = valid.any(dim=2)  # [B, S]，True 表示这一帧有非 (EOS 或 PAD) 的 token
        idx = torch.arange(S, device=x_mel.device).unsqueeze(0).expand(B, S)  # [B, S]
        idx_masked = torch.where(frame_has, idx, torch.full_like(idx, S))  # [B, S]
        first_timestep = idx_masked.min(dim=1).values
    else:
        first_timestep = 1
    if not gt_mel:
        decode_output([x_mel[:, i, :] for i in range(x_mel.shape[1])],
                    f'temp/{model.save_name}/{os.path.basename(midi_path)}_originalmelody.mid', single=True, tempo=90.0)
    x_mel_gt = x_mel.clone()
    x_mel_gt = x_mel_gt[:, first_timestep-1+prompt_length:]
    x_mel = x_mel[:, :prompt_length]
    x_acc = x_acc[:, :prompt_length]
    batch_size, seq_len, subseq_len = x_mel.shape # 10*384*8
    stacked = torch.stack([x_acc, x_mel], dim=2)
    x = stacked.view(batch_size, seq_len * 2, subseq_len)

    if prompt_length!=0:
        decode_output([x[:, i, :] for i in range(x.shape[1])],
                    f'temp/{model.save_name}/{os.path.basename(midi_path)}_promptlen{prompt_length}.mid', tempo=90.0)

    with torch.no_grad():
        x = x.repeat(n_samples, 1, 1)
        x_mel_gt = x_mel_gt.repeat(n_samples, 1, 1)
        import time
        start_time = time.time()
        if prompt_length == 0:
            output = model.global_sampling_from_scratch(x_mel_gt, temperature=temperature, max_seq_len=generation_length)
        else:
            output = model.global_sampling(x, x_mel_gt=x_mel_gt if gt_mel else None, temperature=temperature, max_seq_len=generation_length)
        end_time = time.time()
        print(f"Generation time: {end_time - start_time:.2f} seconds")
    
    for i in range(n_samples):
        output_i = [output[j][i:i + 1, :] for j in range(len(output))]
        decode_output(output_i, f'temp/{model.save_name}/prompt{prompt_length}/{os.path.basename(midi_path)}_temp{temperature}_{i}.mid', tempo=90.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="process midi folder(s) into usable tensors for the task")
    
    parser.add_argument("--model_path", type=str, help="path to model checkpoint")
    parser.add_argument("--prompt_len",type=int, default=75, help="length of prompt")
    parser.add_argument("--n_samples", type=int, default=2, help="number of samples")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature")

    args = parser.parse_args()

    model_path = args.model_path

    if 'small' in model_path:
        model = RoFormerSymbolicTransformer.load_from_checkpoint(model_path, large=False)
    else:
        model = RoFormerSymbolicTransformer.load_from_checkpoint(model_path, large=True)
    model.save_name = os.path.basename(model_path)
    model.cuda()
    model.eval()
    for midi in os.listdir('input/mel'):
        if midi.endswith('mid'):
            midi = os.path.join('input/mel', midi)
            continuation(model, midi, temperature=args.temperature, generation_length=384, n_samples=args.n_samples, prompt_length=args.prompt_len, gt_mel=True)
