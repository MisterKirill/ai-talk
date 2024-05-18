import os
import torch
import librosa
import winsound
from transformers import AutoTokenizer, AutoModelForCausalLM
from obswebsocket import obsws, requests

FIRST_TEXT = 'привет, дурачок'

ws = obsws('localhost', 4455)
ws.connect()

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

def set_subtitle(subtitle: str):
    ws.call(requests.SetInputSettings(
        inputName='Subtitle',
        inputSettings={
            'text': subtitle
        }
    ))

def speak(text: str, speaker: str) -> float:
    audio_paths = tts_model.save_wav(
        text=text,
        speaker=speaker,
        sample_rate=48000
    )
    
    set_subtitle(text)
    
    winsound.PlaySound(audio_paths, winsound.SND_ALIAS)
    
    set_subtitle('')
    
    return librosa.get_duration(path=audio_paths)


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
    
    return context_with_response[0].replace(prompt, '').split('@@ПЕРВЫЙ@@')[0]


def generate_and_speak(speaker: str, text: str) -> str:
    reply = generate_reply(text)
    
    print(f'{speaker}: {reply}')
    
    speak(reply, speaker)
    return reply


def main():
    speaker = 'baya'
    
    reply = generate_and_speak(speaker, FIRST_TEXT)
    
    while True:
        if speaker == 'baya':
            speaker = 'eugene'
        else:
            speaker = 'baya'
        
        reply = generate_and_speak(speaker, reply)


if __name__ == '__main__':
    main()
