import torch
import whisperx
import gc
import os

# Your Hugging Face token
HF_TOKEN = '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'

def transcribe_and_diarize(audio_file_path, number_of_speakers, speaker_names):
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
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=HF_TOKEN,
                device=device
            )
            diarize_segments = diarize_model(
                audio,
                min_speakers=number_of_speakers,
                max_speakers=number_of_speakers
            )

            # Assign speakers to transcription
            result = whisperx.assign_word_speakers(diarize_segments, result)

            # Replace speaker numbers with names
            for segment in result["segments"]:
                if 'speaker' in segment:
                    segment['speaker'] = speaker_names[int(segment['speaker'])]

        except Exception as e:
            print(f"Diarization failed: {str(e)}")
            print("Proceeding without diarization.")

        return result

    except Exception as e:
        print(f"An error occurred during transcription: {str(e)}")
        return None

def save_transcript(result, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for segment in result["segments"]:
            if 'speaker' in segment:
                f.write(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] [{segment['speaker']}] {segment['text']}\n")
            else:
                f.write(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}\n")

def get_wav_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]

def create_transcript_directory(folder_path):
    transcript_dir = os.path.join(folder_path, 'transcripts')
    if not os.path.exists(transcript_dir):
        os.makedirs(transcript_dir)
    return transcript_dir

def collect_speaker_info(wav_files):
    speaker_info = {}
    for wav_file in wav_files:
        print(f"\nEnter speaker information for {wav_file}:")
        num_speakers = int(input("Number of speakers: "))
        speaker_names = []
        for i in range(num_speakers):
            name = input(f"Name of speaker {i + 1}: ")
            speaker_names.append(name)
        speaker_info[wav_file] = {
            'num_speakers': num_speakers,
            'speaker_names': speaker_names
        }
    return speaker_info

def process_folder(folder_path):
    wav_files = get_wav_files(folder_path)
    
    if not wav_files:
        print(f"No WAV files found in {folder_path}")
        return

    print(f"\nProcessing folder: {folder_path}")
    print("WAV files found:")
    for file in wav_files:
        print(f"- {file}")

    speaker_info = collect_speaker_info(wav_files)

    for wav_file in wav_files:
        wav_file_path = os.path.join(folder_path, wav_file)
        print(f"\nProcessing: {wav_file}")

        try:
            num_speakers = speaker_info[wav_file]['num_speakers']
            speaker_names = speaker_info[wav_file]['speaker_names']
            result = transcribe_and_diarize(wav_file_path, num_speakers, speaker_names)

            if result:
                transcript_dir = create_transcript_directory(folder_path)
                transcript_name = f"{os.path.splitext(wav_file)[0]}_transcript.txt"
                output_file = os.path.join(transcript_dir, transcript_name)
                save_transcript(result, output_file)
                print(f"Transcript saved to {output_file}")
            else:
                print("Transcription failed.")
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
