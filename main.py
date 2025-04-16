from datetime import datetime
from flask import Flask, render_template, request, redirect, send_from_directory, flash
import os
import pdfplumber
from google.cloud import texttospeech
from google import genai
from google.genai import types

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
TTS_FOLDER = 'tts'
BOOK_PATH = os.path.join(UPLOAD_FOLDER, 'user_book.pdf')
AUDIO_PATH = os.path.join(UPLOAD_FOLDER, 'audio.wav')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TTS_FOLDER, exist_ok=True)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def synthesize_text(text, output_path):
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)

    with open(output_path, 'wb') as out:
        out.write(response.audio_content)

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    file = request.files['pdf']
    file.save(BOOK_PATH)
    return redirect('/')

@app.route('/ask', methods=['POST'])
def ask():
    # Save audio file
    audio_file = request.files['audio_data']
    audio_file.save(AUDIO_PATH)

    # Step 1: Ask Gemini to transcribe the audio
    client = genai.Client()
    audio_file_ref = client.files.upload(file=AUDIO_PATH)

    transcription_prompt = """
Please provide only the transcription of the user's speech. Do not include any extra commentary or formatting.
Return only the raw text that was spoken.
"""
    transcription_contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(file_uri=audio_file_ref.uri, mime_type="audio/wav"),
                types.Part.from_text(text=transcription_prompt),
            ],
        )
    ]

    transcription_response = client.models.generate_content(
        model="gemini-2.0-pro",
        contents=transcription_contents,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=2048,
        ),
    )

    question_text = transcription_response.text.strip()

    # Step 2: Extract PDF content
    book_text = extract_text_from_pdf(BOOK_PATH)

    # Step 3: Ask Gemini to answer the question based on the book
    qa_prompt = f"""
You are a helpful assistant. The user uploaded the following book:

---BOOK START---
{book_text[:15000]}  # If too large, truncate
---BOOK END---

They asked: "{question_text}"

Answer clearly based only on the book. Keep your answer concise (1–3 sentences).
"""

    answer_response = client.models.generate_content(
        model="gemini-2.0-pro",
        contents=[types.Content(role="user", parts=[types.Part.from_text(qa_prompt)])],
        config=types.GenerateContentConfig(
            temperature=0.5,
            max_output_tokens=512,
        ),
    )

    answer = answer_response.text.strip()

    # Step 4: Synthesize audio response
    output_filename = datetime.now().strftime("%Y%m%d-%H%M%S") + '.mp3'
    output_path = os.path.join(TTS_FOLDER, output_filename)
    synthesize_text(answer, output_path)

    return send_from_directory(TTS_FOLDER, output_filename, mimetype='audio/mp3')

@app.route('/<folder>/<filename>')
def uploaded_file(folder, filename):
    if folder not in ['uploads', 'tts']:
        return "Invalid folder", 404
    path = os.path.join(folder, filename)
    if os.path.exists(path):
        return send_from_directory(folder, filename)
    return "File not found", 404

@app.route('/script.js')
def scripts_js():
    return send_from_directory('static', 'script.js')
