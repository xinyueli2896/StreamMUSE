import os
from mido import MidiFile, MidiTrack, Message, MetaMessage
from tqdm import tqdm

# Input and output directories
INPUT_DIR = '/home/coder/laopo/data/POP909-Dataset/POP909'  # Replace with actual path
OUTPUT_DIR = '/home/coder/laopo/data/POP909-Dataset/mel'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_acc_track(mid_path, output_path):
    midi = MidiFile(mid_path)
    new_midi = MidiFile(ticks_per_beat=midi.ticks_per_beat)

    acc_track = None

    # Step 1: Copy all meta/global tracks (e.g., tempo, time sig)
    for track in midi.tracks:
        new_track = MidiTrack()
        for msg in track:
            if msg.is_meta:
                new_track.append(msg)
        if len(new_track) > 0:
            new_midi.tracks.append(new_track)

    # Step 2: Find and copy the accompaniment track
    for track in midi.tracks:
        # if 'piano' in track.name.lower() or 'bridge' in track.name.lower():
        if 'melody' in track.name.lower():
            acc_track = MidiTrack()
            acc_track.name = track.name  # optional, cosmetic
            for msg in track:
                if not msg.is_meta:
                    acc_track.append(msg)
            # break

    if acc_track:
        new_midi.tracks.append(acc_track)
        new_midi.save(output_path)
        print(f"Saved acc track to: {output_path}")
    else:
        print(f"No acc track found in: {mid_path}")

# Process all MIDI files
for fname in tqdm(os.listdir(INPUT_DIR)):
    if os.path.isdir(os.path.join(INPUT_DIR, fname)):
        input_path = os.path.join(os.path.join(INPUT_DIR, fname), f'{fname}.mid')
        output_path = os.path.join(OUTPUT_DIR, f'{fname}.mid')
        # print(input_path, output_path)
        extract_acc_track(input_path, output_path)
