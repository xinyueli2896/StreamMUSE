import numpy as np

from m2a_transformer_w_chord import RoFormerSymbolicTransformer, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from preprocess_large_midi_dataset import preprocess_midi, DURATION_TEMPLATES
from settings import RWC_DATASET_PATH
import torch
import pretty_midi
import os
import argparse
from rwc_preprocessing import *
from generate_chroma import *
import time
import torch

# def measure_latency(model, x, x_chord, n_warmup=5, n_iters=50):
#     device = next(model.parameters()).device
#     model.eval()
#     x       = x.to(device)
#     x_chord = x_chord.to(device)

#     # warm-up
#     with torch.no_grad():
#         for _ in range(n_warmup):
#             _ = model(x, x_chord)

#     # timed runs
#     times = []
#     use_cuda = device.type == 'cuda'
#     for _ in range(n_iters):
#         if use_cuda: torch.cuda.synchronize()
#         t0 = time.perf_counter()

#         with torch.no_grad():
#             _ = model(x, x_chord)

#         if use_cuda: torch.cuda.synchronize()
#         times.append(time.perf_counter() - t0)

#     avg = sum(times) / len(times)
#     print(f"Avg forward latency: {avg*1000:.2f} ms")
#     return avg

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


def decompress(model, byte_arr_mel, byte_arr_acc, x_chord=None):
    x = torch.tensor(byte_arr_mel).unsqueeze(0)
    x = x.cuda()
    y = torch.tensor(byte_arr_acc).unsqueeze(0)
    y = y.cuda()
    if x_chord!=None:
        x_chord = x_chord.unsqueeze(0).cuda()
        return model.preprocess(x, pitch_shift=torch.zeros(1, dtype=torch.int8).cuda(), x_chord=x_chord, chord_pitch_shift=torch.zeros(1, dtype=torch.int8).cuda(),y=y)
    else:
        return model.preprocess(x, pitch_shift=torch.zeros(1, dtype=torch.int8).cuda(), y=y)

from mido import MidiFile, MidiTrack, MetaMessage, bpm2tempo

def change_tempo(input_path: str, output_path: str, new_bpm: float):
    # Load the original file
    mid = MidiFile(input_path)
    ticks_per_beat = mid.ticks_per_beat

    # Prepare the new file with the same timing resolution
    new_mid = MidiFile()
    new_mid.ticks_per_beat = ticks_per_beat

    # Compute the microseconds per quarter‐note for the target BPM
    new_tempo = bpm2tempo(new_bpm)  # returns µs per beat

    for i, track in enumerate(mid.tracks):
        new_track = MidiTrack()
        # Insert the new set_tempo at time=0 only in the first track
        if i == 0:
            new_track.append(MetaMessage('set_tempo', tempo=new_tempo, time=0))

        # Copy all other messages except any existing set_tempo
        for msg in track:
            if not (msg.is_meta and msg.type == 'set_tempo'):
                new_track.append(msg)
        new_mid.tracks.append(new_track)

    # Save out the tempo‐shifted file
    new_mid.save(output_path)
    print(f"Converted {input_path} → {output_path} @ {new_bpm} BPM")

def continuation(model, midi_path, prompt_length=100, generation_length=384, temperature=1.0, n_samples=1, gt_mel=True, pad_acc=False):
    
    chord_lab_path = '/home/coder/laopo/data/POP909-Dataset/POP909/' + os.path.basename(midi_path)[:3] + '/chord_midi.txt'
    x_chord =  create_chord_chroma(midi_path, chord_lab_path,  subbeat_div=4, shift=0)
    chroma_to_midi(x_chord, f'temp/{model.save_name}/{os.path.basename(midi_path)}_chordgt.mid', base_pitch=48)
    x_mel, x_acc, x_chord = decompress(model, preprocess_midi(midi_path, 4)[0], preprocess_midi(midi_path.replace('mel', 'acc'), 4)[0],x_chord)
    # print("0", x_chord[:,20])
    # print("1", x_chord[:,21])
    # print("10",x_chord[:,100])
    # print("11",x_chord[:,101])
    # if prompt_length==0:
    #     B, S, L = x_mel.shape
    #     valid = (x_mel != EOS_TOKEN) & (x_mel != PAD_TOKEN)  # [B, S, L]，True 表示该 token 有效
    #     frame_has = valid.any(dim=2)  # [B, S]，True 表示这一帧有非 (EOS 或 PAD) 的 token
    #     idx = torch.arange(S, device=x_mel.device).unsqueeze(0).expand(B, S)  # [B, S]
    #     idx_masked = torch.where(frame_has, idx, torch.full_like(idx, S))  # [B, S]
    #     first_timestep = idx_masked.min(dim=1).values
    # else:
    first_timestep = 1
    if not gt_mel:
        decode_output([x_mel[:, i, :] for i in range(x_mel.shape[1])],
                    f'temp/{model.save_name}/{os.path.basename(midi_path)}_originalmelody.mid', single=True, tempo=120.0)
    x_mel_gt = x_mel.clone()
    x_mel_gt = x_mel_gt[:, first_timestep-1+prompt_length:]
    x_chord_gt = x_chord[:, first_timestep-1+prompt_length:]

    x_mel = x_mel[:, :prompt_length] 
    if not pad_acc:
        x_acc = x_acc[:, :prompt_length]
    else:
        x_acc = torch.tensor([3203, 3204, 3204, 3204 , 3204, 3204, 3204, 3204])
        x_acc = x_acc.repeat(1, prompt_length, 1).cuda()

    batch_size, seq_len, subseq_len = x_mel.shape # 10*384*8
    stacked = torch.stack([x_acc, x_mel], dim=2)
    x = stacked.view(batch_size, seq_len * 2, subseq_len)

    if prompt_length!=0:
        decode_output([x[:, i, :] for i in range(x.shape[1])],
                    f'temp/{model.save_name}/{os.path.basename(midi_path)}_promptlen{prompt_length}_{pad_acc}.mid', tempo=120.0)
    
    with torch.no_grad():
        x = x.repeat(n_samples, 1, 1)
        x_mel_gt = x_mel_gt.repeat(n_samples, 1, 1)
        x_chord = x_chord.repeat(n_samples, 1, 1)
        x_chord_gt=x_chord_gt.repeat(n_samples, 1, 1)
        # if prompt_length == 0:
        #     output = model.global_sampling_with_mel_only(x_mel_gt, x_chord_gt, temperature=temperature, max_seq_len=generation_length, prompt_len=75)
        # else:
        output = model.global_sampling(x, x_chord = x_chord, x_mel_gt=x_mel_gt if gt_mel else None, temperature=temperature, max_seq_len=generation_length, prompt_len=prompt_length)

    for i in range(n_samples):
        output_i = [output[j][i:i + 1, :] for j in range(len(output))]
        decode_output(output_i, f'temp/{model.save_name}/prompt{prompt_length}_{pad_acc}/{os.path.basename(midi_path)}_temp{temperature}_{i}.mid', tempo=120)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="process midi folder(s) into usable tensors for the task")
    
    parser.add_argument("--model_path", type=str, help="path to model checkpoint")
    parser.add_argument("--prompt_len",type=int, default=75, help="length of prompt")
    parser.add_argument("--n_samples", type=int, default=2, help="number of samples")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature")
    parser.add_argument("--pad_acc", action="store_true", default=False, help="if with chord")
    args = parser.parse_args()

    model_path = args.model_path

    if 'small' in model_path:
        model = RoFormerSymbolicTransformer.load_from_checkpoint(model_path, large=False)
    else:
        model = RoFormerSymbolicTransformer.load_from_checkpoint(model_path, large=True)
    model.save_name = os.path.basename(model_path)
    model.cuda()
    model.eval()
    for midi in os.listdir('/home/coder/laopo/StreamMUSE/input/mel'):
        if midi.endswith('.mid'):
            midi = os.path.join('/home/coder/laopo/StreamMUSE/input/mel', midi)
            continuation(model, midi, temperature=args.temperature, generation_length=384, n_samples=args.n_samples, prompt_length=args.prompt_len, gt_mel=True,pad_acc=args.pad_acc)
