from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import os
import base64
from google.cloud import texttospeech
from google import genai
from google.genai import types

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('tts', exist_ok=True)

pdf_uri = None  # Global state to store uploaded PDF URI temporarily

@app.route('/upload_text', methods=['POST'])
def upload_text():
    text = request.form['text']
    if not text.strip():
        flash("Text input is empty")
        return redirect(request.url)

    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)

    response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)

    tts_folder = 'tts'
    filename = datetime.now().strftime("%Y%m%d-%I%M%S%p") + '.wav'
    file_path = os.path.join(tts_folder, filename)
    with open(file_path, 'wb') as out:
        out.write(response.audio_content)

    transcript_path = file_path + '.txt'
    with open(transcript_path, 'w') as f:
        f.write(f"Original TTS Input:\n{text}")

    return redirect('/')

def generate(filename=None, prompt=None, file_uri=None):
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    parts = []
    if file_uri:
        parts.append(types.Part.from_uri(file_uri=file_uri, mime_type="application/pdf"))
    elif filename:
        uploaded_file = client.files.upload(file=filename)
        parts.append(types.Part.from_uri(file_uri=uploaded_file.uri, mime_type=uploaded_file.mime_type))

    if prompt:
        parts.append(types.Part.from_text(text=prompt))

    contents = [types.Content(role="user", parts=parts)]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="text/plain",
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=generate_content_config,
    )

    return response.text

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_files(folder):
    return sorted(
        [f for f in os.listdir(folder) if allowed_file(f)],
        reverse=True
    )

@app.route('/')
def index():
    files = get_files(UPLOAD_FOLDER)
    tts_files = get_files('tts')
    return render_template('index.html', files=files, tts_files=tts_files)

@app.route('/upload', methods=['POST'])
def upload_audio():
    global pdf_uri

    if 'pdf_file' in request.files:
        pdf_file = request.files['pdf_file']
        if pdf_file and allowed_file(pdf_file.filename):
            path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
            pdf_file.save(path)
            # Upload to Gemini and store URI
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            uploaded_file = client.files.upload(file=path)
            pdf_uri = uploaded_file.uri

    elif 'audio_data' in request.files:
        file = request.files['audio_data']
        if file.filename:
            filename = datetime.now().strftime("%Y%m%d-%I%M%S%p") + '.wav'
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            prompt = "Please answer this question using the content of the previously uploaded PDF."
            transcript_and_response = generate(filename=path, prompt=prompt, file_uri=pdf_uri)

            # Save transcript
            with open(path + '.txt', 'w') as f:
                f.write(transcript_and_response)

            # Text to Speech
            client_tts = texttospeech.TextToSpeechClient()
            tts_input = texttospeech.SynthesisInput(text=transcript_and_response)
            voice = texttospeech.VoiceSelectionParams(language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
            tts_response = client_tts.synthesize_speech(input=tts_input, voice=voice, audio_config=audio_config)

            tts_path = os.path.join('tts', filename)
            with open(tts_path, 'wb') as f:
                f.write(tts_response.audio_content)

            with open(tts_path + '.txt', 'w') as f:
                f.write(transcript_and_response)

    return redirect('/')

@app.route('/<folder>/<filename>')
def uploaded_file(folder, filename):
    if folder not in ['uploads', 'tts']:
        return "Invalid folder", 404
    folder_path = os.path.join(folder, filename)
    if os.path.exists(folder_path):
        return send_from_directory(folder, filename)
    return "File not found", 404

@app.route('/script.js', methods=['GET'])
def scripts_js():
    return send_from_directory('', 'script.js')

if __name__ == '__main__':
    app.run(debug=True)
