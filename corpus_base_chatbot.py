import docx
import re
import nltk
from nltk.corpus import stopwords
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('omw-1.4')
import PyPDF2
import pycountry
import os
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from collections import Counter
from collections import OrderedDict
import pandas as pd
from nltk.chat.util import Chat, reflections
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from nltk.stem import PorterStemmer
from pattern.text.en import singularize
ps = PorterStemmer()
wnlemmatizer = nltk.stem.WordNetLemmatizer()

def summary(sample):
  parser = PlaintextParser.from_string(sample,Tokenizer("english"))
  summarizer = TextRankSummarizer()
  summary =summarizer(parser.document,2)
  text_summary=""
  for sentence in summary:
      text_summary+=str(sentence)
  return text_summary

def abstract(text):
  if ("abstract" in text):
    idx1 = text.index("abstract")
    if ("keywords" in text):
      idx2 = text.index("keywords") 
    else:
      if "introduction" in text:
        idx2 = text.index("introduction") 
      else:
        idx2 = idx1+200
  else:
    if "introduction" not in text:
      return "Not Found"
    idx1 = text.index("introduction")
    idx2 = idx1+200

  abstract = ''
  for idx in range(idx1 + len("abstract") + 1, idx2):
    abstract = abstract + text[idx]
  return abstract

def talkabout(text):
  notneeded=["scope","research","outlook","techniques","technique","directions","direction","location" ,"author","talks","talk","whose","abstract","summary","talking","problem","statement","tools","tool","technologies","technology","approaches","approach","future","work","works"]
  data = text.translate(str.maketrans('', '', string.punctuation))
  data = data.split() 
  data = list(set(data) - set(notneeded))
  data = [singularize(plural) for plural in data]
  stop_words = set(stopwords.words('english'))
  filtered_sentence = []
  for w in data:
    if w not in stop_words:
        filtered_sentence.append(w)
  word_counts = Counter(filtered_sentence)
  most_occur = word_counts.most_common(2)
  talking = "The paper is mostly talking about "+ most_occur[0][0]+" , "+most_occur[1][0]
  return word_counts,talking

def author(text):
  idx1 = 0
  if "Abstract" in text:
    idx1 = text.index("Abstract")
  elif "ABSTRACT" in text:
    idx1 = text.index("ABSTRACT")
  elif "Introduction" in text:
    idx1 = text.index("Introduction")
  elif "INTRODUCTION" in text:
    idx1 = text.index("INTRODUCTION")
  abstract = ''
  for idx in range(0, idx1):
    abstract = abstract + text[idx]

  Author={}
  nltk_results = ne_chunk(pos_tag(word_tokenize(abstract)))
  for nltk_result in nltk_results:
      if type(nltk_result) == Tree:
          name = ''
          for nltk_result_leaf in nltk_result.leaves():
              name += nltk_result_leaf[0] + ' '
          Author[name] = nltk_result.label() 
  if ('PERSON' in Author.values()):
    value = [i for i in Author if Author[i]=="PERSON"]
    values=""
    count=0 
    while(len(value)>0 and count<3):
      values+=(value.pop())
      values+=", "
      count+=1
    return values
  else:
    value = list(Author.keys())
    if len(value)>0:
      return value.pop()
    else:
      return "Not Found"

def authorlocation(text):
  idx1 = 0
  if "Abstract" in text:
    idx1 = text.index("Abstract")
  elif "ABSTRACT" in text:
    idx1 = text.index("ABSTRACT")
  elif "Introduction" in text:
    idx1 = text.index("Introduction")
  elif "INTRODUCTION" in text:
    idx1 = text.index("INTRODUCTION")
  abstract = ''
  for idx in range(0, idx1):
    abstract = abstract + text[idx]
  countries = []
  for country in pycountry.countries:
      if country.name in abstract or  country.alpha_3 in abstract:
          countries.append(country.name)
  if(len(countries)>0):
    return countries.pop()
  else:
    return "Not Found"

def tools(text):
  techniques=''
  tech=[]
  tokens=nltk.word_tokenize(text)
  for i in range(len(tokens)):
    if (tokens[i]=="tools" or tokens[i]=="tool" ):
      if (tokens[i-1]==")"):
        tech.append(tokens[i-3])
        tech.append(tokens[i-2])
        tech.append(tokens[i-1])
      else:
        tech.append(tokens[i-1])
      for j in range(i,i+10):
        tech.append(tokens[j])
      for k in range(i+11,len(tokens)):
        tech.append(tokens[k])
        if(tokens[k]=='.'):
          break
      tech.append("\n")
  if(len(tech)>0):
    return " ".join(tech)
  else:
    return "Not Found"

def techniques(text):
  techniques=''
  tech=[]
  tokens=nltk.word_tokenize(text)
  for i in range(len(tokens)):
    if (tokens[i]=="techniques" or tokens[i]=="technique" or tokens[i]=="technology" or tokens[i]=="technologies" or tokens[i]=="approach" or tokens[i]=="approaches"  ):
      if (tokens[i-1]==")"):
        tech.append(tokens[i-3])
        tech.append(tokens[i-2])
        tech.append(tokens[i-1])
      else:
        tech.append(tokens[i-1])
      for j in range(i,i+15):
        tech.append(tokens[j])
      for k in range(i+16,len(tokens)):
        tech.append(tokens[k])
        if(tokens[k]=='.'):
          break
      tech.append("\n")
    elif(tokens[i]=="references" or tokens[i]=="reference"):
      break
  if(len(tech)>0):
    return " ".join(tech)
  else:
    return "Not Found"

def problem(text):
  if ("problem statement" in text):
    idx1 = text.index("problem statement")
    idx2 = idx1+500
    future=''
    for idx in range(idx1, idx2):
      future = future + text[idx]
    while(text[idx2]!='.'):
      future = future + text[idx2]
      idx2+=1
    return future
  elif ("introduction" in text):
    idx1 = text.index("introduction")
    idx2 = idx1+1000
    future=''
    for idx in range(idx1+len("introduction")+1, idx2):
      future = future + text[idx]
    while(text[idx2]!='.'):
      future = future + text[idx2]
      idx2+=1
    return summary(future)
  else:
    problem = abstract(text)
    return summary(problem)

def problemdiscuss(text):
  if ("problem" in text):
    idx1 = text.index("problem")
    idx2 = idx1+500
    future=''
    for idx in range(idx1, idx2):
      future = future + text[idx]
    while(text[idx2]!='.'):
      future = future + text[idx2]
      idx2+=1
    return future
  elif ("conclusion" in text):
    idx1 = text.index("conclusion")
    idx2 = idx1+1000
    future=''
    for idx in range(idx1+len("conclusion")+1, idx2):
      future = future + text[idx]
    while(text[idx2]!='.'):
      future = future + text[idx2]
      idx2+=1
    return summary(future)
  else:
    problem = abstract(text)
    return summary(problem)

def conclusion(text):
  tokens=nltk.word_tokenize(text)
  word_counts = Counter(tokens)
  if(word_counts['conclusion']>1):
    while(word_counts['conclusion']!=1):
      tokens.remove("conclusion")
      word_counts = Counter(tokens)
  text=" ".join(tokens)
  if ("conclusion" in text):
    idx1 = text.index("conclusion")
    if ("references" in text):
      idx2 = text.index("references") 
    elif("reference" in text):
      idx2 = text.index("reference") 
    else:
      idx2 = idx1+1000
    future=''
    for idx in range(idx1+len("conclusion")+1, idx2):
      future = future + text[idx]
    while(text[idx2]!='.'):
      future = future + text[idx2]
      idx2+=1
    return future
  else:
    
    return 'NOT FOUND'

def futurework(text):
  if ("future work" in text):
    idx1 = text.index("future work")
    idx2 = idx1+500
    future=''
    for idx in range(idx1, idx2):
      future = future + text[idx]
    while(text[idx2]!='.'):
      future = future + text[idx2]
      idx2+=1
    return future 
  elif ("future works" in text):
    idx1 = text.index("future works")
    idx2 = idx1+500
    future=''
    for idx in range(idx1, idx2):
      future = future + text[idx]
    while(text[idx2]!='.'):
      future = future + text[idx2]
      idx2+=1
    return future
  elif ("future direction" in text):
    idx1 = text.index("future direction")
    idx2 = idx1+500
    future=''
    for idx in range(idx1, idx2):
      future = future + text[idx]
    while(text[idx2]!='.'):
      future = future + text[idx2]
      idx2+=1
    return future
  elif ("future directions" in text):
    idx1 = text.index("future directions")
    idx2 = idx1+500
    future=''
    for idx in range(idx1, idx2):
      future = future + text[idx]
    while(text[idx2]!='.'):
      future = future + text[idx2]
      idx2+=1
    return future
  else:
    return "Not Found"

def getText(filename):
    if(".pdf" in filename):
      pdfFileObj = open(filename, 'rb')
      #Reading each report pdf 
      pdfReader = PyPDF2.PdfReader(pdfFileObj)
      content = ""
      #Checking if the file is encrypted or not if encrypted then will skip otherwise it will read the content
      #This loop reads the content from the pdf file page wise 
      for i in range(len(pdfReader.pages)):
          pageObj = pdfReader.pages[i]
          if(i != 0):
            content += " "  
          #Extracting text from the pdf
          content += pageObj.extract_text()
      text = re.sub(r'\s+', ' ',re.sub(r'\[[0-9]*\]', ' ', content))
    else:
      doc = docx.Document(filename)
      fullText = []
      for para in doc.paragraphs:
          fullText.append(para.text)
      text = '\n'.join(fullText)
      text = re.sub(r'\s+', ' ',re.sub(r'\[[0-9]*\]', ' ', text))
        
    dataframe = pd.DataFrame(columns=['Questions','Answers'])
    newrow = {"Questions":"who is the author","Answers":author(text)}
    dataframe = dataframe.append(newrow, ignore_index=True)
    newrow = {"Questions":"what is the location of author","Answers":authorlocation(text)}
    dataframe = dataframe.append(newrow, ignore_index=True)
    text = text.lower()
    newrow = {"Questions":"what is the summary","Answers":summary(text)}
    dataframe = dataframe.append(newrow, ignore_index=True)
    newrow = {"Questions":"what is the abstract","Answers":abstract(text)}
    dataframe = dataframe.append(newrow, ignore_index=True)
    newrow = {"Questions":"what is the future work","Answers":futurework(text)}
    dataframe = dataframe.append(newrow, ignore_index=True)
    newrow = {"Questions":"what techniques has been discussed","Answers":techniques(text)}
    dataframe = dataframe.append(newrow, ignore_index=True)
    newrow = {"Questions":"what approaches has been discussed","Answers":techniques(text)}
    dataframe = dataframe.append(newrow, ignore_index=True)
    newrow = {"Questions":"what tools has been suggested","Answers":tools(text)}
    dataframe = dataframe.append(newrow, ignore_index=True)
    newrow = {"Questions":"what is the problem statement","Answers":problem(text)}
    dataframe = dataframe.append(newrow, ignore_index=True)
    newrow = {"Questions":"what is the problem discussed","Answers":problemdiscuss(text)}
    dataframe = dataframe.append(newrow, ignore_index=True)
    newrow = {"Questions":"what is the conclusion","Answers":conclusion(text)}
    dataframe = dataframe.append(newrow, ignore_index=True)
    occurence,talking=talkabout(text)
    newrow = {"Questions":"what is the paper talking about","Answers":talking}
    dataframe = dataframe.append(newrow, ignore_index=True)
    occur = dict(occurence)
    for key,item in occur.items():
      if key!="direction" and key!="location" and key!="author" and key!="summary" and key!="talk" and key!="whose" and key!="abstract" and key!="summary" and key!="talking" and key!="problem" and key!="statement" and key!="tool" and key!="technology" and key!="approach" and key!="approach" and key!="future" and key!="work"  :
        question = "How many times "+str(key)+" has been appeared"
        answer = key +" has been appeared "+str(item)+" times"
        newrow = {"Questions":question,"Answers":answer}
        dataframe = dataframe.append(newrow, ignore_index=True)
    dataframe.to_csv("Rules.csv",index=False)

def perform_lemmatization_stemming(tokens):
    tokens = [wnlemmatizer.lemmatize(token) for token in tokens]
    tokens = [ps.stem(token) for token in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w.lower() in stop_words]
    return tokens
punctuation_removal = dict((ord(punctuation), None) for punctuation in string.punctuation)
def get_processed_text(document):
    return perform_lemmatization_stemming(nltk.word_tokenize(document.lower().translate(punctuation_removal)))

def Chat_Bot_Answer(Question,Chat_Bot_Responses):
    #Append the question to the question defined earlier as rules
    Question_In_Rules=list(Chat_Bot_Responses.keys())
    Question_In_Rules.append(Question)
    #Create the sentence vector based on the questions list
    Vectorizer = TfidfVectorizer()
    Sentences_Vectors = Vectorizer.fit_transform(Question_In_Rules)
    #Measure the Cosine Similarity of the queston ask by user with other questions
    #And take the second closest index because the first index is the user own question
    Values = cosine_similarity(Sentences_Vectors[-1], Sentences_Vectors)
    Closest_Question = Question_In_Rules[Values.argsort()[0][-2]]
    #Final check to make sure there is answer present to the user's question.
    #If all the answers are 0, means the question ask by user are not in the rules.
    Invalid_Response = Values.flatten()
    Invalid_Response.sort()
    if Invalid_Response[-2] == 0:
        return "Sorry! I couldn't understand."
    else: 
        return Chat_Bot_Responses[Closest_Question]

def get_chatbotReply(question):
    question=question.lower()
    #Reading rules for chatbot from the CSV file
    Rules = pd.read_csv("Rules.csv")
    #Building a dictionary where questions are keys and answers are aalues 
    Chat_Bot_Responses = Rules.set_index('Questions').to_dict()['Answers']
    #Converting the questions into lower case
    Chat_Bot_Responses = dict((k.lower(), v) for k, v in Chat_Bot_Responses .items()) 
    Chatdict = {value:key for key, value in Chat_Bot_Responses.items()}
    for key,value in Chatdict.items():
      Chatdict[key] = ' '.join(get_processed_text(value))
    Chat_Bot_Responses = {value:key for key, value in Chatdict.items()}
    pairs = []
    #Converting the dictionary into pairs where each list index and question and answers in it
    for key,item in Chat_Bot_Responses.items():
        value = []
        #Appending the answers into a list
        value.append(item)
        anotherList = []
        #Appending the question and answer in the same list
        anotherList.append(key)
        anotherList.append(value)
        #Appending the list having question adn it's answer to the pairs list
        pairs.append(anotherList)
    
    question = ' '.join(get_processed_text(question))
    #Training the chabot using chat library with the pairs we made from the rules given and the reflections
    chat = Chat(pairs, reflections)
    #Getting one to one response in present it in rules then will give answer
    #Will calculate cosine similarity with all questions and gives answer of the closest question if one to one question doesn't found
    return Chat_Bot_Answer(question,Chat_Bot_Responses)