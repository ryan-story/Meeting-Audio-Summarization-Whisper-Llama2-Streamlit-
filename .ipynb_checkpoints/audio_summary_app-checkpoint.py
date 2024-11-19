import streamlit as st
import tempfile
import os
from pydub import AudioSegment
import warnings
from transformers import pipeline
import subprocess

# Function to convert audio to WAV format
def convert_audio_to_wav(audio_file_path):
    try:
        audio = AudioSegment.from_file(audio_file_path)
        wav_file_path = audio_file_path.split(".")[0] + ".wav"
        audio.export(wav_file_path, format="wav")
        return wav_file_path
    except Exception as e:
        raise ValueError(f"Error converting audio to WAV: {e}")

# Function to transcribe audio using Whisper
def transcribe_audio_with_whisper(audio_file_path):
    result = asr_model(audio_file_path, return_timestamps=True)
    return result['text']

# Ensure Ollama is running
subprocess.run(["ollama", "serve"])
# Function to summarize the transcript using Ollama
def summarize_with_ollama(input_prompt, input_text):
    prompt = f"{input_prompt}\n\n{input_text}"
    result = subprocess.run(
        ["ollama", "run", "llama2"],
        input=prompt,
        text=True,
        capture_output=True
    )
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        raise RuntimeError(f"Error: {result.stderr}")

# Split the transcript into smaller chunks, for LLaMA 2 it is recommended to split chunks based on 2500 words
def chunk_text(text, max_chunk_size=2500):
    words = text.split()
    chunks = [' '.join(words[i:i + max_chunk_size]) for i in range(0, len(words), max_chunk_size)]
    return chunks

# Summarize each chunk and concatenate the results
def summarize_large_transcript(transcript):
    chunks = chunk_text(transcript)
    combined_summary = ""
    for i, chunk in enumerate(chunks):
        st.write(f"Summarizing chunk {i + 1}/{len(chunks)}...")
        partial_summary = summarize_with_ollama("Summarize the following transcript:", chunk)
        combined_summary += f"\n\nChunk {i + 1} Summary:\n{partial_summary}"
    return combined_summary

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize the Whisper model
asr_model = pipeline("automatic-speech-recognition", model="openai/whisper-large", device=0)

# Main Streamlit application
def main():
    st.title("MEETING AUDIO SUMMARY")
    st.write("Upload a meeting audio file to generate derived content.")

    # File uploader
    file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

    if file and 'transcription' not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

        # Get audio transcription
        st.write("Using Whisper to transcribe audio into text...")
        try:
            if file.type == "audio/mpeg":
                temp_wav_path = convert_audio_to_wav(temp_file_path)
                transcription = transcribe_audio_with_whisper(temp_wav_path)
                os.remove(temp_wav_path)
            else:
                transcription = transcribe_audio_with_whisper(temp_file_path)

            os.remove(temp_file_path)

            # Save the transcription and chunked summary to session state
            st.session_state['transcription'] = transcription
            st.session_state['chunked_summary'] = summarize_large_transcript(transcription)

            st.write("Transcription and Summaries Generated.")

        except ValueError as e:
            st.error(f"Audio processing error: {e}")
        except RuntimeError as e:
            st.error(f"Summary generation error: {e}")

    # Check if chunked_summary is already in session state
    if 'chunked_summary' in st.session_state:

        # Prompt user for custom input after chunked summary is created
        user_prompt = st.text_input(
            "Enter your custom prompt to generate final content:",
            "Summarize the following transcript and include an overall summary, action items, and follow-ups:"
        )

        # Generate final summary when the user enters a prompt
        if user_prompt:
            st.write("Generating requested content...")
            try:
                final_summary = summarize_with_ollama(user_prompt, st.session_state['chunked_summary'])
                st.write("Requested Content:")
                st.write(final_summary)

                # Download the final summary
                st.download_button(
                    label="Download Content",
                    data=final_summary,
                    file_name="content.txt",
                    mime="text/plain"
                )
                st.download_button(
                    label="Download Transcript",
                    data=st.session_state['transcription'],
                    file_name="transcript.txt",
                    mime="text/plain"
                )
            except RuntimeError as e:
                st.error(f"Summary generation error: {e}")

if __name__ == "__main__":
    main()
