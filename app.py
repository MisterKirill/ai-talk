import os
import torch
import librosa
import winsound
from flask import Flask, request
from transformers import AutoTokenizer, AutoModelForCausalLM
from threading import Thread

device = torch.device('cpu')
torch.set_num_threads(4)
local_file = 'model.pt'

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file(
        'https://models.silero.ai/models/tts/ru/v4_ru.pt',
        local_file
    )  

tts_model = torch.package.PackageImporter(local_file).load_pickle('tts_models', 'model')
tts_model.to(device)

tokenizer = AutoTokenizer.from_pretrained('tinkoff-ai/ruDialoGPT-medium')
model = AutoModelForCausalLM.from_pretrained('tinkoff-ai/ruDialoGPT-medium')

app = Flask(__name__)

def speak(text: str, speaker: str) -> float:
    audio_paths = tts_model.save_wav(
        text=text,
        speaker=speaker,
        sample_rate=48000
    )
    
    Thread(target=play_speech, args=(audio_paths,), daemon=True).start()
    
    return librosa.get_duration(path=audio_paths)

def play_speech(audio_paths: str):
    winsound.PlaySound(audio_paths, winsound.SND_ASYNC | winsound.SND_ALIAS)

def generate_reply(text: str) -> str:
    prompt = f'@@ПЕРВЫЙ@@{text}@@ВТОРОЙ@@'
    
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
    
    print(context_with_response)
    
    return context_with_response[0].replace(prompt, '').split('@@ПЕРВЫЙ@@')[0]

@app.route('/generate', methods=['GET'])
def generate():
    speaker = request.args.get('speaker')
    text = request.args.get('text')
    
    if not speaker or not text:
        return 'Bad Request', 400
    
    reply = generate_reply(text)
    duration = speak(reply, speaker)
    
    return {
        'reply': reply,
        'duration': duration
    }
