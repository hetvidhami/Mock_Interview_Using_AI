import whisper
import subprocess as sp
import os
import re
from datetime import datetime

# Initialize the Whisper model
model = whisper.load_model("base")

# Directory for input and output files
input_audio_file = "input1.mp4"  # Path to your input audio
output_dir = "output"  # Directory to store the split audio parts
os.makedirs(output_dir, exist_ok=True)

# Function to split the audio using FFmpeg
def split_audio(input_file, output_dir, segment_duration=30):
    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Get the duration of the audio file (in seconds)
    cmd = ['ffmpeg', '-i', input_file, '-f', 'null', '-']
    result = sp.run(cmd, stderr=sp.PIPE, text=True)
    duration_line = [line for line in result.stderr.splitlines() if "Duration" in line]
    if duration_line:
        duration_str = re.search(r'Duration: (\d+):(\d+):(\d+)\.(\d+)', duration_line[0])
        hours, minutes, seconds = int(duration_str.group(1)), int(duration_str.group(2)), int(duration_str.group(3))
        total_duration = hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError("Could not fetch audio duration.")
    
    # Split the audio into smaller parts based on segment duration
    num_segments = total_duration // segment_duration + 1
    for i in range(num_segments):
        start_time = i * segment_duration
        output_file = os.path.join(output_dir, f"{base_name}_part_{i + 1}.mp3")
        cmd = ['ffmpeg', '-i', input_file, '-ss', str(start_time), '-t', str(segment_duration), '-vn', '-acodec', 'copy', output_file]
        sp.run(cmd)
        print(f"Segment {i + 1} saved as: {output_file}")
    return num_segments

# Function to transcribe the audio using Whisper
def transcribe_audio(audio_file):
    print(f"Transcribing {audio_file}...")
    result = model.transcribe(audio_file)
    return result['text']

# Split the audio into parts and transcribe each part
num_parts = split_audio(input_audio_file, output_dir)

# Process each split audio file and transcribe
transcriptions = []
for i in range(num_parts):
    audio_file = os.path.join(output_dir, f"your_audio_part_{i + 1}.mp4")
    text = transcribe_audio(audio_file)
    transcriptions.append(text)

# Combine all transcriptions
full_transcription = "\n".join(transcriptions)
print("\nFull Transcription:")
print(full_transcription)

# Optional: Save the transcription to a text file
with open('full_transcription.txt', 'w') as f:
    f.write(full_transcription)
    print("Transcription saved to full_transcription.txt")
