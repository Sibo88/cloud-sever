from flask import Flask, request, jsonify
import os
import wave
import json
import numpy as np
from vosk import Model, KaldiRecognizer
import language_tool_python
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
model_path = "models/vosk-model-en-us-0.42-gigaspeech"
model = Model(model_path)
tool = language_tool_python.LanguageTool('en-US')

@app.route('/upload', methods=['POST'])
def upload_audio():
    file = request.files['audio']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Transcribe audio
    text = transcribe(filepath)
    # Grammar check
    matches = tool.check(text)
    grammar_issues = [{"message": m.message, "offset": m.offset, "error": m.context} for m in matches]
    # NLP feedback
    feedback = analyze_text(text)

    return jsonify({
        "transcription": text,
        "grammar_issues": grammar_issues,
        "repetitive_words": feedback["repetitive"],
        "filler_words": feedback["filler"],
        "total_words": feedback["total_words"]
    })

def transcribe(wav_path):
    wf = wave.open(wav_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())

    result_text = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            result_text += result.get("text", "") + " "

    final_result = json.loads(rec.FinalResult())
    result_text += final_result.get("text", "")
    return result_text.strip()

def analyze_text(text):
    words = text.split()
    total = len(words)
    word_count = {}
    for word in words:
        word = word.lower().strip(".,!?;:\"'()[]{}")
        word_count[word] = word_count.get(word, 0) + 1

    repetitive = {w: c for w, c in word_count.items() if c > 1}
    filler_words = {"uh", "um", "so", "because"}
    filler = {w: word_count.get(w, 0) for w in filler_words if w in word_count}
    return {
        "repetitive": repetitive,
        "filler": filler,
        "total_words": total
    }

if __name__ == '__main__':
    app.run(debug=True)
