from flask import Flask, request, render_template
from model import get_answer, generate_questions

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    paragraph = request.form['paragraph']
    num_questions = 5  # Number of questions to generate
    questions = generate_questions(paragraph, num_questions)
    result = ""
    for question in questions:
        answer = get_answer(question, paragraph)
        result += f'<strong>Question:</strong> {question}<br><strong>Answer:</strong> {answer}<br><br>'
    return result

if __name__ == '__main__':
    app.run(debug=True)
