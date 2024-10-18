import json
import re

def parse_time(time_str):
    return float(time_str[:-1])  # Remove 's' and convert to float

def convert_transcript(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    segments = []
    speakers = set()
    max_time = 0

    for line in lines:
        match = re.match(r'\[(\d+\.\d+s) -> (\d+\.\d+s)\] \[Speaker (SPEAKER_\d+)\] (.*)', line.strip())
        if match:
            start_time, end_time, speaker, text = match.groups()
            start = parse_time(start_time)
            end = parse_time(end_time)
            speakers.add(speaker)
            max_time = max(max_time, end)

            segments.append({
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text.strip()
            })

    transcript = {
        "metadata": {
            "duration": max_time,
            "speakers": list(speakers)
        },
        "segments": segments
    }

    with open(output_file, 'w') as file:
        json.dump(transcript, file, indent=2)

    print(f"Conversion complete. JSON file saved as {output_file}")

# Usage
input_file = 'audio_transcript.txt'  # Replace with your input file name
output_file = 'transcript.json'

convert_transcript(input_file, output_file)
