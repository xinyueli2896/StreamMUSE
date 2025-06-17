import numpy as np
import xf_midi
from settings import RWC_DATASET_PATH, LA_DATASET_PATH, NOTTINGHAM_DATASET_PATH
import os
from joblib import Parallel, delayed
import torch
import shutil
import json
import pretty_midi
import argparse

tokenize_dict = {'<sos>': 0, '<eos>': 1, '<pad>': 2}
tokenize_count = [-1, -1, -1]

DURATION_TEMPLATES = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096])

def update_token_dict(midi_path, beat_div=4):
    try:
        midi = xf_midi.XFMidi(midi_path, constant_tempo=60.0 / beat_div)
    except:
        return None
    midi_end_time = int(midi.get_end_time())
    if midi_end_time <= 0:
        return
    duration_boundaries = (DURATION_TEMPLATES[1:] + DURATION_TEMPLATES[:-1]) / 2
    for ins in midi.instruments:
        program = ins.program
        if ins.is_drum:
            program = 127
        for note in ins.notes:
            start_time = int(round(note.start))
            end_time = int(round(note.end))
            if start_time >= 0 and end_time < midi_end_time:
                if ins.is_drum:
                    duration = 0
                else:
                    duration = np.searchsorted(duration_boundaries, end_time - start_time)
                key = (program, note.pitch, duration)
                if key not in tokenize_dict:
                    tokenize_dict[key] = len(tokenize_dict)
                    tokenize_count.append(0)
                tokenize_count[tokenize_dict[key]] += 1

def preprocess_midi(midi_path, max_polyphony, beat_div=4, ins_ids='all'):
    # print(midi_path)
    try:
        midi = xf_midi.XFMidi(midi_path, constant_tempo=60.0 / beat_div)
    except Exception as e:
        print(f"Error processing {midi_path}: Invalid MIDI file. Error: {e}")
        return None
    # print(midi)
    midi_end_time = int(midi.get_end_time())
    # print("midi_end_time:",midi_end_time)
    # print(midi_end_time)
    if midi_end_time <= 0:
        return None
    if not isinstance(ins_ids, list):
        ins_ids = [ins_ids]
    duration_boundaries = (DURATION_TEMPLATES[1:] + DURATION_TEMPLATES[:-1]) / 2
    # print(duration_boundaries)
    min_pitch = 127
    max_pitch = 0
    result_rolls = []
    # print(ins_ids)
    for ins_id in ins_ids:
        has_any_note = False
        rolls = np.full((midi_end_time, max_polyphony, 3), dtype=np.uint8, fill_value=255)
        polyphony_counts = np.zeros(midi_end_time, dtype=np.uint8)
        for i, ins in enumerate(midi.instruments):
            # print(i, ins)
            program = ins.program
            if ins.is_drum:
                program = 127
            for note in ins.notes:
                start_time = int(round(note.start))
                end_time = int(round(note.end))
                if start_time >= 0 and end_time < midi_end_time and polyphony_counts[start_time] < max_polyphony:
                    if ins.is_drum:
                        duration = 0
                    else:
                        duration = np.searchsorted(duration_boundaries, end_time - start_time) #为了做四舍五入的quantize
                        min_pitch = min(min_pitch, note.pitch)
                        max_pitch = max(max_pitch, note.pitch)
                    add_note = False
                    if ins_id == 'all':
                        add_note = True
                    elif isinstance(ins_id, int):
                        raise NotImplementedError
                    elif isinstance(ins_id, str):
                        if '-' in ins_id:
                            task, num = ins_id.split('-')
                            num = int(num)
                            if task == 'track':
                                add_note = i == num
                            elif task == 'upto':
                                add_note = i <= num
                            elif task == 'from':
                                add_note = i >= num
                            elif task == 'notrack':
                                add_note = i != num
                            else:
                                raise NotImplementedError
                        elif ins_id == 'drum':
                            add_note = ins.is_drum
                        elif ins_id == 'nondrum':
                            add_note = not ins.is_drum
                        elif ins_id == 'empty':
                            add_note = False
                        else:
                            raise NotImplementedError
                    else:
                        raise NotImplementedError
                    if add_note:
                        has_any_note = True
                        rolls[start_time, polyphony_counts[start_time]] = [program, note.pitch, duration]
                        # [program, pitch, duration]
                        polyphony_counts[start_time] += 1
        if not has_any_note and ins_id != 'empty':
            return None  # invalid midi file
        for i in range(midi_end_time):
            # Sort notes by ins first, then by pitch, then by duration
            rolls[i, :polyphony_counts[i]] = rolls[i, :polyphony_counts[i]][np.lexsort((rolls[i, :polyphony_counts[i], 2], rolls[i, :polyphony_counts[i], 1], rolls[i, :polyphony_counts[i], 0]))]
            if polyphony_counts[i] < max_polyphony:
                rolls[i, polyphony_counts[i], 0] = 254  # EOS token
        result_rolls.append(rolls)
    result_rolls = np.concatenate(result_rolls, axis=1)
    # print(result_rolls)
    # Get song-level pitch shift range
    pitch_shift_max = 127 - max_pitch
    pitch_shift_min = -min_pitch
    # print(midi_path, ": final", torch.tensor(result_rolls.reshape(midi_end_time, -1)).shape, torch.tensor([pitch_shift_min, pitch_shift_max], dtype=torch.int8).shape)
    return torch.tensor(result_rolls.reshape(midi_end_time, -1)), torch.tensor([pitch_shift_min, pitch_shift_max], dtype=torch.int8)

def tensor_to_midi(
    rolls: torch.LongTensor,
    save_path: str,
    tempo: float = 120.0,
    instrument_program: int = 0,
):
    """
    rolls: [T, 3*max_polyphony]  each row is
      [ prog0, pitch0, dur0,  prog1, pitch1, dur1,  ... ]
      prog==254 -> EOS, pitch==255 -> PAD  
    tempo: playback tempo in BPM  
    instrument_program: GM program number (0=Steinway Grand Piano)
    """
    # 1) set up PrettyMIDI + one piano instrument
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=instrument_program, is_drum=False)
    pm.instruments.append(piano)

    # 2) compute real-time per time-step
    time_step = 60.0 / tempo / 4

    T, C = rolls.shape
    max_poly = C // 3

    # 3) iterate each time step
    for t in range(T):
        row = rolls[t]
        for slot in range(max_poly):
            base = slot * 3
            prog  = int(row[base].item())
            pitch = int(row[base + 1].item())
            dur_i = int(row[base + 2].item())

            # end-of-sequence → stop this frame
            # if prog == tokenize_dict['<eos>']:
            #     break
            # padding → skip
            if pitch == 255:
                continue
            # sanity check
            if not (0 <= pitch < 128):
                continue
            if not (0 <= dur_i < len(DURATION_TEMPLATES)):
                continue

            start = t * time_step
            end   = start + DURATION_TEMPLATES[dur_i] * time_step
            piano.notes.append(
                pretty_midi.Note(
                    velocity=100,
                    pitch=pitch,
                    start=start,
                    end=end
                )
            )

    # 4) write file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pm.write(save_path)

def create_tokenize_dict(folder):
    # Get all midi files in the folder, recursively
    midi_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.mid') or file.endswith('.MID'):
                midi_files.append(os.path.join(root, file))
    from tqdm import tqdm
    for i, midi_file in enumerate(tqdm(midi_files)):
        update_token_dict(midi_file)
        if i % 100 == 0:
            print('Tokenize dict:', len(tokenize_dict))
    print('Tokenize dict:', len(tokenize_dict))

def create_npy_dataset_from_midi(folders,
                                 max_polyphony,
                                 dataset_name,
                                 ins_ids='all',
                                 scan_subfolders=True,
                                 max_idx=None):

    # 1. 收集所有 .mid/.MID 文件
    midi_files = []
    for folder in folders:
        for filename in os.listdir(folder):
            if filename.lower().endswith('.mid'):
                midi_files.append(os.path.join(folder, filename))
    midi_files = sorted(midi_files)
    if max_idx is not None:
        midi_files = midi_files[:max_idx]
    print(f'Processing {len(midi_files)} files')

    # 2. 定义一个 worker：对每个 midi_file 调用 preprocess_midi，并捕获异常
    def safe_preprocess(midi_path):
        try:
            # 这里原本是 preprocess_midi 返回 (data, shift)
            data, shift = preprocess_midi(midi_path, max_polyphony, ins_ids=ins_ids)
            return (midi_path, data, shift, None)
        except Exception as e:
            # 出错时返回错误信息
            return (midi_path, None, None, str(e))

    # 3. 并行执行
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(safe_preprocess)(midi_path) for midi_path in midi_files
    )

    # 4. 拆分成功与失败
    successful_data = []
    successful_shifts = []
    lengths = []
    failed_list = []  # 用于记录出错的文件及错误信息

    for midi_path, data, shift, error in results:
        if error is None:
            # 正常返回的数据 data: Tensor([…]), shift: Tensor([…])
            successful_data.append(data)
            successful_shifts.append(shift)
            lengths.append(len(data))
        else:
            # 记录出错的文件和对应的错误堆栈/消息
            failed_list.append((midi_path, error))

    # 5. 如果需要看到哪些文件处理失败，可以打印或写入一个日志文件
    if failed_list:
        print("----- 以下文件处理失败：-----")
        for path, err_msg in failed_list:
            print(f"{path}  →  错误信息: {err_msg}")
        print("----------------------------")

    # 6. 只有当有“成功”结果时，才做拼接与保存
    if successful_data:
        all_data = torch.cat(successful_data, dim=0)
        all_shifts = torch.cat(successful_shifts, dim=0)
        # 保存到 disk
        torch.save(all_data, f'data/{dataset_name}.pt')
        torch.save(all_shifts, f'data/{dataset_name}.pitch_shift_range.pt')
        torch.save(torch.tensor(lengths), f'data/{dataset_name}.length.pt')
    else:
        print("没有成功的文件，跳过保存。")

    return [f'data/{dataset_name}.pt', f'data/{dataset_name}.length.pt']

def sync(path_m, path_a):

    tensor_m = path_m[0]
    length_m = path_m[1]

    tensor_a = path_a[0]
    length_a = path_a[1]


    mel = torch.load(tensor_m)
    acc = torch.load(tensor_a)

    length_mel = torch.load(length_m)
    length_acc = torch.load(length_a)

    min_lengths = torch.minimum(length_mel, length_acc)

    adjusted_tensor1 = []
    adjusted_tensor2 = []
    start_idx1 = 0
    start_idx2 = 0

    for i in range(len(min_lengths)):
        # Trim subtensors to min length
        adjusted_tensor1.append(mel[start_idx1:start_idx1 + min_lengths[i]])
        adjusted_tensor2.append(acc[start_idx2:start_idx2 + min_lengths[i]])

        # Update indices
        start_idx1 += length_mel[i]
        start_idx2 += length_acc[i]

    # Convert back to tensors
    adjusted_tensor1 = torch.cat(adjusted_tensor1)
    adjusted_tensor2 = torch.cat(adjusted_tensor2)


    torch.save(min_lengths, length_m)
    torch.save(min_lengths, length_a)
   
    torch.save(adjusted_tensor1, tensor_m)
    torch.save(adjusted_tensor2, tensor_a)
    print(adjusted_tensor1.shape)
    print(adjusted_tensor2.shape)
    print('tensor matching complete')

def create_tensor(dataset_name, folders, max_polyphony=4):
    paths = create_npy_dataset_from_midi(folders, max_polyphony, f'{dataset_name}_cp{max_polyphony}', scan_subfolders=False)
    return paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="process midi folder(s) into usable tensors for the task")
    
    # Positional arguments
    parser.add_argument("--name",type=str,help="name your dataset")
    parser.add_argument("--folders",nargs='+', type=str, help="paths to midi folders")
    parser.add_argument("--polyphony",type=int,default=4,help="maximum number of notes allowed in one timestep")

    args = parser.parse_args()

    melody_f = [os.path.join(f, 'mel') for f in args.folders]
    accomp_f = [os.path.join(f, 'acc') for f in args.folders]
    melody_n = args.name+'_mel'
    accomp_n = args.name+'_acc'

    paths_m = create_tensor(melody_n, melody_f, max_polyphony=args.polyphony)
    paths_a = create_tensor(accomp_n, accomp_f, max_polyphony=args.polyphony)
    
    sync(paths_m, paths_a)


        

    