from rwc_preprocessing import *
from tqdm import tqdm
def chroma_to_midi(chroma, output_path, frame_dur=0.125, base_pitch=60):
    """
    chroma: np.ndarray (n_frames,12), values 0 or 1
    output_path: where to save the .mid
    frame_dur: seconds per frame (0.125 for 1/16-note @120 BPM)
    base_pitch: MIDI number to add to pitch-class (e.g. 60)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0)  # default piano

    n_frames = chroma.shape[0]
    prev = None
    seg_start = 0

    # walk through frames + a dummy end frame
    for i in range(n_frames + 1):
        curr = None if i == n_frames else tuple(chroma[i])
        if curr != prev:
            # flush the segment [seg_start, i)
            if prev is not None:
                t0 = seg_start * frame_dur
                t1 = i         * frame_dur
                for pc, on in enumerate(prev):
                    if on:
                        note = pretty_midi.Note(
                            velocity=100,
                            pitch=base_pitch + pc,
                            start=t0,
                            end=t1
                        )
                        inst.notes.append(note)
            seg_start = i
            prev = curr

    pm.instruments.append(inst)
    pm.write(output_path)
    print(f"Wrote {output_path} with {len(inst.notes)} notes.")

# test

# midi_path = '/home/coder/laopo/data/POP909-Dataset/mel/005.mid'
# chord_lab_path = '/home/coder/laopo/data/POP909-Dataset/POP909/005/chord_midi.txt'
# output_path = 'new.mid'
# chroma = create_chord_chroma(midi_path, chord_lab_path, subbeat_div=4, shift=0)
# print(chroma.shape)
# chroma_to_midi(chroma, output_path)

# process

# midis = os.listdir('/home/coder/laopo/data/POP909-Dataset/mel')
# midis = [os.path.join('/home/coder/laopo/data/POP909-Dataset/mel',i) for i in midis]
# midis = sorted(midis)
# chroma_list = []
# length_list = []
# c=0
# for midi_path in tqdm(midis):
#     chord_lab_path = '/home/coder/laopo/data/POP909-Dataset/POP909/' + os.path.basename(midi_path)[:3] + '/chord_midi.txt'
#     chroma = create_chord_chroma(midi_path, chord_lab_path, subbeat_div=4, shift=0)
#     chroma_list.append(chroma)
#     length_list.append(chroma.shape[0])
#     # if c==2: break
#     c+=1
# chroma_tensor = torch.cat(chroma_list, dim=0)
# print(chroma_tensor.shape)
# length_tensor = torch.tensor(length_list)
# print(length_tensor)
# torch.save(chroma_tensor, '/home/coder/laopo/StreamMUSE/data/909_chord.pt')
# torch.save(length_tensor, '/home/coder/laopo/StreamMUSE/data/909_chord.length.pt')
# length = torch.load('/home/coder/laopo/StreamMUSE/data/909_cp4_v2_mel.length.pt')
# print(length)

# sync

# mel = torch.load('/home/coder/laopo/StreamMUSE/data/909correct_mel_cp4.pt')
# acc = torch.load('/home/coder/laopo/StreamMUSE/data/909correct_acc_cp4.pt')
# chord = torch.load('/home/coder/laopo/StreamMUSE/data/909_chord.pt')

# length_mel = torch.load('/home/coder/laopo/StreamMUSE/data/909correct_mel_cp4.length.pt')
# length_chord = torch.load('/home/coder/laopo/StreamMUSE/data/909_chord.length.pt')

# min_lengths = torch.minimum(length_mel, length_chord)

# adjusted_tensor1 = []
# adjusted_tensor2 = []
# adjusted_tensor3 = []
# start_idx1 = 0
# start_idx2 = 0

# for i in range(len(min_lengths)):
#     # Trim subtensors to min length
#     adjusted_tensor1.append(mel[start_idx1:start_idx1 + min_lengths[i]])
#     adjusted_tensor2.append(acc[start_idx1:start_idx1 + min_lengths[i]])
#     adjusted_tensor3.append(chord[start_idx2:start_idx2 + min_lengths[i]])

#     # Update indices
#     start_idx1 += length_mel[i]
#     start_idx2 += length_chord[i]

# # Convert back to tensors
# adjusted_tensor1 = torch.cat(adjusted_tensor1)
# adjusted_tensor2 = torch.cat(adjusted_tensor2)
# adjusted_tensor3 = torch.cat(adjusted_tensor3)


# torch.save(min_lengths, '/home/coder/laopo/StreamMUSE/data/909correct_mel_cp4.length.pt')
# torch.save(min_lengths, '//home/coder/laopo/StreamMUSE/data/909correct_acc_cp4.length.pt')
# torch.save(min_lengths, '/home/coder/laopo/StreamMUSE/data/909_chord.length.pt')

# torch.save(adjusted_tensor1, '/home/coder/laopo/StreamMUSE/data/909correct_mel_cp4.pt')
# torch.save(adjusted_tensor2, '/home/coder/laopo/StreamMUSE/data/909correct_acc_cp4.pt')
# torch.save(adjusted_tensor3,'/home/coder/laopo/StreamMUSE/data/909_chord.pt')
# # print(min_lengths)
# print(adjusted_tensor1.shape)
# print(adjusted_tensor2.shape)
# print(adjusted_tensor3.shape)
# print('tensor matching complete')
