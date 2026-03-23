from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

app = Flask(__name__)

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "AventIQ-AI/t5-stockmarket-qa-chatbot"
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = T5Tokenizer.from_pretrained(model_name)

@app.route('/ask', methods=['POST'])
def ask():
    # Get question from the frontend
    question = request.json.get('question', '')
    input_text = "question: " + question
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=50)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Return the answer as a JSON response
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
