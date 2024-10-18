import torch
import whisperx
import gc
import os
import random
from pyannote.audio import Pipeline
from pydub import AudioSegment
from pydub.playback import play

# Your Hugging Face token
HF_TOKEN = 'enter hugging face token here'

def transcribe_and_diarize(audio_file_path, hf_token):
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"

    try:
        # Load audio file
        audio = whisperx.load_audio(audio_file_path)

        # Transcribe with WhisperX
        model = whisperx.load_model("large-v2", device, compute_type=compute_type)
        result = model.transcribe(audio, batch_size=16)

        # Align whisper output
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=device
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False
        )

        # Free up memory
        del model
        del model_a
        gc.collect()
        torch.cuda.empty_cache()

        # Diarization
        try:
            # Initialize the pipeline
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                                use_auth_token=hf_token)

            # Apply the pipeline to the audio file
            diarization = pipeline(audio_file_path)

            # Extract speaker segments
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker
                })

            # Assign speakers to transcription segments
            for segment in result['segments']:
                segment_mid = (segment['start'] + segment['end']) / 2
                for speaker_segment in speaker_segments:
                    if speaker_segment['start'] <= segment_mid <= speaker_segment['end']:
                        segment['speaker'] = speaker_segment['speaker']
                        break

            # Count unique speakers
            unique_speakers = set(segment['speaker'] for segment in result['segments'] if 'speaker' in segment)
            num_speakers = len(unique_speakers)

            print(f"Detected {num_speakers} unique speakers in the audio.")

        except Exception as e:
            print(f"Diarization failed: {str(e)}")
            print("Proceeding without diarization.")
            num_speakers = 0

        return result, num_speakers

    except Exception as e:
        print(f"An error occurred during transcription: {str(e)}")
        return None, 0

def get_speaker_samples(result, num_samples=3):
    speaker_samples = {}
    for segment in result['segments']:
        if 'speaker' in segment:
            if segment['speaker'] not in speaker_samples:
                speaker_samples[segment['speaker']] = []
            speaker_samples[segment['speaker']].append(segment)
    
    # Select random samples for each speaker
    for speaker in speaker_samples:
        if len(speaker_samples[speaker]) > num_samples:
            speaker_samples[speaker] = random.sample(speaker_samples[speaker], num_samples)
    
    return speaker_samples

def play_audio_segment(audio_file, start_time, end_time):
    try:
        audio = AudioSegment.from_wav(audio_file)
        segment = audio[start_time*1000:end_time*1000]
        play(segment)
    except Exception as e:
        print(f"Failed to play audio segment: {str(e)}")
        print("Continuing without audio playback.")

def user_assign_speaker_names(result, audio_file_path):
    speaker_samples = get_speaker_samples(result)
    speaker_names = {}

    print("\nPlease identify the speakers based on the following samples:")
    for speaker, samples in speaker_samples.items():
        print(f"\nSpeaker {speaker}:")
        for sample in samples:
            print(f"  {sample['text']}")
            play_audio_segment(audio_file_path, sample['start'], sample['end'])
        
        name = input(f"Enter name for {speaker}: ").strip()
        speaker_names[speaker] = name

    # Confirm speaker names
    print("\nConfirm speaker names:")
    for speaker, name in speaker_names.items():
        confirm = input(f"Is '{name}' correct for {speaker}? (y/n): ").strip().lower()
        if confirm != 'y':
            new_name = input(f"Enter new name for {speaker}: ").strip()
            speaker_names[speaker] = new_name

    # Update the result with assigned names
    for segment in result['segments']:
        if 'speaker' in segment and segment['speaker'] in speaker_names:
            segment['speaker'] = speaker_names[segment['speaker']]

    return result, speaker_names

def save_transcript(result, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for segment in result["segments"]:
            speaker = segment.get('speaker', 'Unknown')
            f.write(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] [{speaker}] {segment['text']}\n")

def get_wav_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]

def create_transcript_directory(folder_path):
    transcript_dir = os.path.join(folder_path, 'transcripts')
    if not os.path.exists(transcript_dir):
        os.makedirs(transcript_dir)
    return transcript_dir

def process_audio_file(audio_file_path, hf_token, output_file):
    result, num_speakers = transcribe_and_diarize(audio_file_path, hf_token)
    
    if result:
        print(f"\nTranscription completed. Detected {num_speakers} speakers.")
        result, speaker_names = user_assign_speaker_names(result, audio_file_path)
        save_transcript(result, output_file)
        print(f"\nTranscript saved to {output_file}")
        print("Identified speakers:", ', '.join(speaker_names.values()))
    else:
        print("Transcription failed.")

def process_folder(folder_path):
    wav_files = get_wav_files(folder_path)
    
    if not wav_files:
        print(f"No WAV files found in {folder_path}")
        return

    print(f"\nProcessing folder: {folder_path}")
    print("WAV files found:")
    for file in wav_files:
        print(f"- {file}")

    for wav_file in wav_files:
        wav_file_path = os.path.join(folder_path, wav_file)
        print(f"\nProcessing: {wav_file}")

        try:
            transcript_dir = create_transcript_directory(folder_path)
            transcript_name = f"{os.path.splitext(wav_file)[0]}_transcript.txt"
            output_file = os.path.join(transcript_dir, transcript_name)
            process_audio_file(wav_file_path, HF_TOKEN, output_file)
        except Exception as e:
            print(f"An error occurred while processing {wav_file}: {str(e)}")

def main():
    cwd = os.getcwd()
    directories = [d for d in os.listdir(cwd) if os.path.isdir(os.path.join(cwd, d))]

    if not directories:
        print("No directories found in the current working directory.")
        return

    for directory in directories:
        process_folder(os.path.join(cwd, directory))

    print("\nAll directories processed.")

if __name__ == "__main__":
    main()
