from flask import Flask,flash, request, redirect, url_for
from flask import render_template
from flask import request
from flask import jsonify
from flask_cors import CORS, cross_origin
import corpus_base_chatbot as chats
from werkzeug.utils import secure_filename
import os


UPLOAD_FOLDER = os.getcwd()+'/'

app = Flask(__name__)
CORS(app, support_credentials=True)#import templates.chat_bot as chat
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
flag = True

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/index')
def upload():
    return render_template('index.html')

@app.route('/chosecsv',methods=["POST"])
def chosecsv():
    file = request.files['file']
    if file.filename == '':
        return "Fail"
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    chats.getText(filename)
    return "Success"

@app.route('/response',methods=["POST"])
def get_response():
    data = request.form['usermessage']
    reply = chats.get_chatbotReply(data)
    print(reply)
    return reply


if __name__ == "__main__":
    app.run(debug=True,port=3000)