import os
from mido import MidiFile, MidiTrack, Message
from tqdm import tqdm
import shutil

# Input and output directories
INPUT_DIR = 'POP909-Dataset/POP909'          # Replace with actual path
OUTPUT_DIR = 'datasets/Processed-POP909-Dataset'          # Where to save the melody-only MIDI files
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_acc_track(mid_path, output_path):
    midi = MidiFile(mid_path)
    for i, track in enumerate(midi.tracks):
        if 'piano' in track.name.lower() or 'bridge' in track.name.lower():
            new_midi = MidiFile()
            new_track = MidiTrack()
            new_midi.tracks.append(new_track)
            for msg in track:
                new_track.append(msg)
            new_midi.save(output_path)
            print(f"Saved acc track to: {output_path}")
            return
    print(f"No acc track found in: {mid_path}")

def process_POP909_dataset(input_dir, output_dir):
    output_mel_dir = os.path.join(output_dir, 'mel')
    output_acc_dir = os.path.join(output_dir, 'acc')
    os.makedirs(output_mel_dir, exist_ok=True)
    os.makedirs(output_acc_dir, exist_ok=True)
    for dirname in tqdm(os.listdir(input_dir)):
        dir_path = os.path.join(input_dir, dirname)
        if os.path.isdir(dir_path):
            for filename in os.listdir(dir_path):
                if filename.endswith('.mid'):
                    mid_path = os.path.join(dir_path, filename)
                    output_mel_path = os.path.join(output_mel_dir, f"{filename}")
                    output_acc_path = os.path.join(output_acc_dir, f"{filename}")
                    
                    # Extract melody track
                    extract_acc_track(mid_path, output_acc_path)

                    # Copy the original MIDI file to the melody directory
                    shutil.copy(mid_path, output_mel_path)
                    print(f"Copied melody track to: {output_mel_path}")


if __name__ == "__main__":
    process_POP909_dataset(INPUT_DIR, OUTPUT_DIR)
    
    
    