# Raza Report

## About

This repo contains a Streamlit app. The Streamlit app allows dragging and dropping MP3 files of audio. It converts the speech to text using Whisper, and then summarizes the text into meeting notes using Llama2.

## Dependencies
- python 3.7 or above (not in requirements.txt)
- Ollama
- Git

## Instructions
1. Ensure python is installed
2. Ensure Git is installed
3. Install Ollama by visiting https://ollama.com/download
4. Ensure Ollama is in your Applications directory
5. Manually launch Ollama and continue through prompts until you click "Finish"
6. Clone this repository to your intended working directory
7. In Terminal, "ollama pull llama2"
8. In Terminal, navigate to your workingh directory
9. In Terminal, "pip install -r requirements.txt"
10. In Terminal, "streamlit run raza_report_app.py"

## Limitations
This application is designed to run on a local machine, therefore running large models locally requires sufficient CPU/GPU and memory resources.