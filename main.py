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


#Home API shows upload paper option when application is run
@app.route('/')
def home():
    return render_template('upload.html')

#Index API shows chatbot interface when user upload the paper and submit it
@app.route('/index')
def upload():
    return render_template('index.html')

#Chose CSV API is called when user submit the paper this api first of all move the uploaded file from its 
#original directory to the current working directory and then it extract the text from the paper uploaded by calling the function
#getText from the corpus base chatbot python file 
@app.route('/chosecsv',methods=["POST"])
def chosecsv():
    file = request.files['file']
    if file.filename == '':
        return "Fail"
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    chats.getText(filename)
    return "Success"


#Resposne API is called when user send a message to chatbot it will get chatbot response by calling the function
#get_chatbotReply from the corpus base chatbot python file for the question user asked 
@app.route('/response',methods=["POST"])
def get_response():
    data = request.form['usermessage']
    reply = chats.get_chatbotReply(data)
    print(reply)
    return reply


if __name__ == "__main__":
    app.run(debug=True,port=3000)