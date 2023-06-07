from flask import Flask, render_template, request
from flask_sock import Sock
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('tinkoff-ai/ruDialoGPT-medium')
model = AutoModelForCausalLM.from_pretrained('tinkoff-ai/ruDialoGPT-medium')

app = Flask(__name__, static_folder = "static")
sock = Sock(app)

@app.route("/")
def index():
    return render_template('index.html')

# API
@app.route("/api/generate", methods = ["GET"])
def generate_reply():
    prompt = request.args.get("p")
    if not prompt:
        return "fatal error 000000000000000000000x1"
    
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
    
    if not reply.find('@@ВТОРОЙ@@') == -1:
        reply = reply.split('@@ВТОРОЙ@@')[0]
    
    return reply
