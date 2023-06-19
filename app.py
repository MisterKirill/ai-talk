from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForCausalLM
from threading import Thread
from num2words import num2words
import wave
import pyaudio
import os
import torch
import re

print('Initializing ruDialoGPT...')

tokenizer = AutoTokenizer.from_pretrained('tinkoff-ai/ruDialoGPT-medium')
model = AutoModelForCausalLM.from_pretrained('tinkoff-ai/ruDialoGPT-medium')

print('Initializing TTS...')

device = torch.device('cpu')
torch.set_num_threads(4)
local_file = 'tts_model.pt'

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v3_1_ru.pt', local_file)  

tts_model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
tts_model.to(device)

sample_rate = 48000

app = Flask(__name__, static_folder = "static")

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/api/generate", methods = ["GET"])
def generate_reply():
    prompt = request.args.get("p")
    if not prompt:
        return "fatal error 000000000000000000000x1"
    
    speaker = request.args.get("speaker")
    if not speaker:
        return "fatal error 000000000000000000000x2"
    
    if speaker == 'man':
        speaker = 'eugene'
    elif speaker == 'woman':
        speaker = 'kseniya'
        
    inputs = tokenizer(prompt, return_tensors='pt')
    generated_token_ids = model.generate(
        **inputs,
        top_k=10,
        top_p=0.95,
        num_beams=3,
        num_return_sequences=3,
        do_sample=True,
        no_repeat_ngram_size=2,
        temperature=1.2,
        repetition_penalty=1.2,
        length_penalty=1.0,
        eos_token_id=50257,
        max_new_tokens=40
    )
    
    context_with_response = [tokenizer.decode(sample_token_ids) for sample_token_ids in generated_token_ids]
    reply = context_with_response[0].replace(prompt, "").split("@@ПЕРВЫЙ@@")[0]
    
    # Extract numbers and convert them to words
    numbers = re.findall(r'\d+', reply)
    for number in numbers:
        wordsnum = num2words(int(number), lang='ru')
        reply = reply.replace(number, wordsnum)
    
    audio_paths = tts_model.save_wav(text=reply, speaker=speaker, sample_rate=sample_rate)
    wf = wave.open(audio_paths)
    
    # Get duration of wav
    frames = wf.getnframes()
    duration = frames / float(sample_rate)
    
    thread = Thread(target=play_sound, args=(wf,))
    thread.start()
    
    return {
        'duration': duration,
        'reply': reply
    }

def play_sound(wf):
    p = pyaudio.PyAudio()
    chunk = 1024
    
    stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
        channels = wf.getnchannels(),
        rate = wf.getframerate(),
        output = True)
    
    data = wf.readframes(chunk)
    while data:
        stream.write(data)
        data = wf.readframes(chunk)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
