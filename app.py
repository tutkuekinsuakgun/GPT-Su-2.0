import os
from flask import Flask, request, jsonify, render_template
from model import generate_response

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    input_text = request.form['input_text']
    generated_text = generate_response(input_text)
    return jsonify({'generated_text': generated_text})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

