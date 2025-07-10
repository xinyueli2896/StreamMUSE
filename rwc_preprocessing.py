from settings import RWC_DATASET_PATH, LA_DATASET_PATH
import os
import xf_midi
import pretty_midi
import numpy as np
import json
from tqdm import tqdm
np.int = int
from mir_eval.chord import encode, rotate_bitmap_to_root
import torch
def chord_to_midi(chord_str, bass_starting_pitch=36):
    root_number, semitone_bitmap, bass_number = encode(chord_str)
    if root_number < 0:
        return []
    bass_number = (bass_number + root_number) % 12
    semitone_bitmap = rotate_bitmap_to_root(semitone_bitmap, root_number)
    pitches = [bass_number + bass_starting_pitch]
    _, extended_semitone_bitmap, _ = encode(chord_str, reduce_extended_chords=True)
    extended_semitone_bitmap = rotate_bitmap_to_root(extended_semitone_bitmap, root_number)
    for i in range(12):
        if semitone_bitmap[i]:
            pitches.append(i + bass_starting_pitch + 12)
        elif extended_semitone_bitmap[i]:
            pitches.append(i + bass_starting_pitch + 24)
    return pitches

def create_drum_track(beat_time, downbeat_time, subbeat_time_boundaries, unit_time):
    drum_track = pretty_midi.Instrument(program=0, is_drum=True)
    def quantize_time(time):
        return np.searchsorted(subbeat_time_boundaries, time)
    quantized_downbeats = quantize_time(downbeat_time)
    start_id = quantized_downbeats[0]
    if quantized_downbeats[2] - quantized_downbeats[1] != quantized_downbeats[1] - quantized_downbeats[0]:
        start_id = quantized_downbeats[1]
    quantized_beats = np.arange(len(subbeat_time_boundaries) + 1)
    quantized_stronger_beats = quantize_time(np.interp(np.arange(len(downbeat_time) - 1) + 0.5, np.arange(len(downbeat_time)), downbeat_time))
    downbeat_ins = pretty_midi.DRUM_MAP.index('Acoustic Bass Drum') + 35
    stronger_beat_ins = pretty_midi.DRUM_MAP.index('Acoustic Snare') + 35
    beat_ins = pretty_midi.DRUM_MAP.index('Closed Hi Hat') + 35
    for t in quantized_downbeats:
        if t >= start_id:
            drum_track.notes.append(pretty_midi.Note(
                velocity=100,
                pitch=downbeat_ins,
                start=t * unit_time,
                end=(t + 1) * unit_time
            ))
    for t in quantized_stronger_beats:
        if t >= start_id:
            drum_track.notes.append(pretty_midi.Note(
                velocity=100,
                pitch=stronger_beat_ins,
                start=t * unit_time,
                end=(t + 1) * unit_time
            ))
    for t in quantized_beats:
        if t >= start_id:
            drum_track.notes.append(pretty_midi.Note(
                velocity=100,
                pitch=beat_ins,
                start=t * unit_time,
                end=(t + 1) * unit_time
            ))
    return drum_track

def add_chord_track(midi_path, chord_lab_path, output_path, subbeat_div=2, shift=0):
    performance_midi = pretty_midi.PrettyMIDI(midi_path)
    score_midi = xf_midi.XFMidi(midi_path, constant_tempo=120.0)
    beat_time = performance_midi.get_beats()
    downbeat_time = performance_midi.get_downbeats()
    # interpolate to get subbeat time
    n_beats = len(beat_time)
    subbeat_indices = np.arange((n_beats - 1) * subbeat_div + 1) / subbeat_div
    subbeat_time = np.interp(subbeat_indices, np.arange(n_beats), beat_time)
    subbeat_time_boundaries = (subbeat_time[:-1] + subbeat_time[1:]) / 2
    chord_track = pretty_midi.Instrument(program=0, name='chord')
    drum_track = create_drum_track(beat_time, downbeat_time, subbeat_time_boundaries, 60.0 / 120 / subbeat_div)
    drum_track.name = 'drum'
    if chord_lab_path is not None:
        f = open(chord_lab_path, 'r')
        lines = [line.strip() for line in f.readlines() if line.strip()]
        f.close()
    else:
        lines = []  # no chord labels
    def quantize_time(time):
        return np.searchsorted(subbeat_time_boundaries, time)
    quantized_downbeats = quantize_time(downbeat_time)
    quantized_downbeats = np.concatenate([[-np.inf], quantized_downbeats, [np.inf]])
    for line in lines:
        # print(line)
        start_time, end_time, chord = line.split('\t')
        start_time = float(start_time)
        end_time = float(end_time)
        start_time_quantized = quantize_time(start_time)
        end_time_quantized = quantize_time(end_time)
        pitches = chord_to_midi(chord)
        # separate chords at downbeats
        start_downbeat_id = np.searchsorted(quantized_downbeats, start_time_quantized)
        end_downbeat_id = np.searchsorted(quantized_downbeats, end_time_quantized)
        for downbeat_id in range(start_downbeat_id, end_downbeat_id + 1):
            s = max(start_time_quantized, quantized_downbeats[downbeat_id - 1])
            e = min(end_time_quantized, quantized_downbeats[downbeat_id])
            if s >= e:
                continue
            for pitch in pitches:
                chord_track.notes.append(pretty_midi.Note(
                    velocity=100,
                    pitch=pitch,
                    start=s / (subbeat_div * 2),
                    end=e / (subbeat_div * 2),
                ))
    if shift != 0:
        for ins in score_midi.instruments:
            for note in ins.notes:
                note.start += shift / 2
                note.end += shift / 2
    score_midi.instruments.insert(0, drum_track)
    score_midi.instruments.insert(0, chord_track)
    score_midi.write(output_path)

def create_rwc_chord_dataset():
    chord_lab_path = os.path.join(RWC_DATASET_PATH, 'MidiAlignedChord')
    output_path = os.path.join('temp', 'rwc_chord')
    for file in os.listdir(chord_lab_path):
        if file.endswith('.TXT'):
            midi_path = os.path.join(RWC_DATASET_PATH, 'AIST.RWC-MDB-P-2001.SMF_SYNC', file[:-15] + '.SMF_SYNC.MID')
            add_chord_track(midi_path, os.path.join(chord_lab_path, file), os.path.join(output_path, file[:-15] + '.mid'))
def pitches_to_chroma(pitches, normalize=False):
    chroma = np.zeros(12, dtype=np.float32)
    for p in pitches:
        pc = p % 12
        chroma[pc] = 1

    chroma = torch.tensor(chroma, dtype=torch.uint8)
    return chroma
def create_chord_chroma(midi_path, chord_lab_path,  subbeat_div=4, shift=0):
    midi = pretty_midi.PrettyMIDI(midi_path)
    beat_path = os.path.join(f"/home/coder/laopo/data/POP909-Dataset/POP909/{os.path.basename(midi_path)[:3]}", "beat_midi.txt")
    print(f"reading from {beat_path}")
    f = open(beat_path, 'r')
    lines = [line.strip() for line in f.readlines() if line.strip()]
    f.close()

    beat_time = [float(i.split(' ')[0]) for i in lines]
    y = np.arange(len(beat_time)) * subbeat_div

    def performance_to_score(perf_time):
        return np.interp(perf_time, beat_time, y)
    # score_midi = xf_midi.XFMidi(midi_path, constant_tempo=60/subbeat_div)
    # midi_end_time = int(score_midi.get_end_time())
    midi_end_time = int(performance_to_score(midi.get_end_time()))

    # print(midi_end_time)
    # beat_time = performance_midi.get_beats()
    # print(beat_time)
    # downbeat_time = performance_midi.get_downbeats()
    downbeat_time = beat_time[::2]
    # print(downbeat_time)
    # interpolate to get subbeat time
    n_beats = len(beat_time)
    subbeat_indices = np.arange((n_beats - 1) * subbeat_div + 1) / subbeat_div
    subbeat_time = np.interp(subbeat_indices, np.arange(n_beats), beat_time)
    subbeat_time_boundaries = (subbeat_time[:-1] + subbeat_time[1:]) / 2
    chord_track = pretty_midi.Instrument(program=0)
    drum_track = create_drum_track(beat_time, downbeat_time, subbeat_time_boundaries, 60.0 / 120 / subbeat_div)
    if chord_lab_path is not None:
        f = open(chord_lab_path, 'r')
        lines = [line.strip() for line in f.readlines() if line.strip()]
        f.close()
    else:
        lines = []  # no chord labels
    def quantize_time(time):
        return np.searchsorted(subbeat_time_boundaries, time)
    quantized_downbeats = quantize_time(downbeat_time)
    quantized_downbeats = np.concatenate([[-np.inf], quantized_downbeats, [np.inf]])
    chromas = torch.zeros((midi_end_time, 12), dtype=torch.uint8)
    # print(chromas)
    for line in lines:
        start_time, end_time, chord = line.split('\t')
        start_time = float(start_time)
        end_time = float(end_time)
        # print(start_time)
        start_time_quantized = quantize_time(start_time)
        end_time_quantized = quantize_time(end_time)
        pitches = chord_to_midi(chord)
        chroma = pitches_to_chroma(pitches)
        # separate chords at downbeats
        start_downbeat_id = np.searchsorted(quantized_downbeats, start_time_quantized)
        end_downbeat_id = np.searchsorted(quantized_downbeats, end_time_quantized)
        for downbeat_id in range(start_downbeat_id, end_downbeat_id + 1):
            s = max(start_time_quantized, quantized_downbeats[downbeat_id - 1])
            e = min(end_time_quantized, quantized_downbeats[downbeat_id])
            if s >= e:
                continue
            # print(s, e)    
            chromas[int(s):int(e), :] = chroma
    return chromas
def find_files_starting_with(prefix, directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(prefix) and file.endswith('.lab'):
                return os.path.join(root, file)
import mido

def extract_tracks(input_file, output_melody_file, output_chord_file):
    # Load the MIDI file
    midi = mido.MidiFile(input_file)

    # Create new MIDI files for melody and chords
    melody_midi = mido.MidiFile()
    chord_midi = mido.MidiFile()

    # Create new tracks for each
    melody_track = mido.MidiTrack()
    chord_track = mido.MidiTrack()

    melody_midi.tracks.append(melody_track)
    chord_midi.tracks.append(chord_track)

    # Loop through all tracks to identify melody and chord tracks
    for track in midi.tracks:
        if track.name.lower().startswith('mel') or track.name.lower().startswith('voc'):  # Adjust condition as needed
            melody_track.extend(track)
        if "chord" in track.name.lower():  # Adjust condition as needed
            chord_track.extend(track)

    # Save the extracted tracks into separate files
    melody_midi.save(output_melody_file)
    chord_midi.save(output_chord_file)
    print(f"Melody saved to {output_melody_file}")
    print(f"Chords saved to {output_chord_file}")

if __name__ == '__main__':
    # midi_path = '/home/coder/laopo/data/rwc/mid_with_chord'
    # melody_out = '/home/coder/laopo/data/rwc/mel_mid'
    # chord_out = '/home/coder/laopo/data/rwc/chord_mid'
    # for i in tqdm(os.listdir(midi_path)):
    #     i_ = os.path.join(midi_path,i)
    #     mel_output = os.path.join(melody_out, i)
    #     chord_output = os.path.join(chord_out, i)
    #     extract_tracks(i_, mel_output, chord_output)
    # test_songs = [
    #     ['394045b83f247bb862d7b09b1aacd78f.mid', 0],
    #     ['4261342f0970488e1381cb39867c48e1.mid', 0],
    #     ['f947e58c78aa7c8055ef8dfc424ca22e.mid', 0],
    #     ['d9520bbf2bccd6424aa09f5694aa68f7.mid', 4.0],
    #     ['cf5f3bc804e474f4d0baf0c74656b042.mid', 1.0],
    # ]
    # for test_song, shift in test_songs:
    #     file_path = os.path.join(LA_DATASET_PATH, 'MIDIs', test_song[0], test_song)
    #     output_path = os.path.join('temp', 'la_beat', test_song.replace('.mid', '_beat.mid'))
    #     add_chord_track(file_path, None, output_path, shift=shift)



    bad_list = []
    midi_path = '/home/coder/laopo/data/rwc/AIST.RWC-MDB-P-2001.SMF_SYNC'
    chord_path = '/home/coder/laopo/data/rwc/MidiAlignedChord'
    for i in tqdm(os.listdir(midi_path)):
        if not i.endswith('.MID'):
            print(f'skipping {i}')
            continue
        try:
            inx = i.split('-')[1][1:4]
        except:
            print(f"something wrong with {i}")
        
        chord_p = os.path.join(chord_path, f"RM-P{inx}.MIDI.CHORD.TXT")
        output_p = os.path.join('/home/coder/laopo/data/rwc/chord_mid_aligned', f"RM-P{inx}.MIDI.CHORD.MID")
        # if os.path.exists(output_p):
        #     print(f"skipping {inx}")
        #     continue
        print(f'processing {chord_p}')
        midi_p = os.path.join(midi_path,i)
        try:
            add_chord_track(midi_p, chord_p, output_p)
        except Exception as e:
            bad_list.append(inx)
            print(e)

    file_path = "badlist.json"

    # Write the list to a JSON file
    with open(file_path, "w") as json_file:
        json.dump(bad_list, json_file, indent=4)

    print("failed midi write to list")
                
