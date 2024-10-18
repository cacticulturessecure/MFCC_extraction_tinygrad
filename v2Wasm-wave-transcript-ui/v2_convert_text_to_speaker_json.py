import json
import re

def parse_time(time_str):
    return float(time_str[:-1])  # Remove 's' and convert to float

def convert_transcript(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
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

    # Prompt user for custom speaker names
    speaker_mapping = {}
    for speaker in speakers:
        custom_name = input(f"Enter a custom name for {speaker}: ").strip()
        speaker_mapping[speaker] = custom_name if custom_name else speaker

    # Replace speaker names in segments
    for segment in segments:
        segment["speaker"] = speaker_mapping[segment["speaker"]]

    transcript = {
        "metadata": {
            "duration": max_time,
            "speakers": list(speaker_mapping.values())
        },
        "segments": segments
    }

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(transcript, file, indent=2, ensure_ascii=False)

    print(f"Conversion complete. JSON file saved as {output_file}")
    print("Custom speaker names:")
    for original, custom in speaker_mapping.items():
        print(f"{original} -> {custom}")

# Usage
input_file = 'audio_transcript.txt'  # Replace with your input file name
output_file = 'transcript.json'

convert_transcript(input_file, output_file)
