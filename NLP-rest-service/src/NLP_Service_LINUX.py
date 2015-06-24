#!flask/bin/python
from flask import Flask, jsonify,abort,send_from_directory
from flask import request
import os
import pickle 
import nltk
import string
from itertools import chain
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.classify import NaiveBayesClassifier as nbc
from nltk.corpus import CategorizedPlaintextCorpusReader
from nltk.tag.stanford import POSTagger
import re
from nltk.stem import SnowballStemmer
from nltk.classify.util import accuracy
import cherrypy
from paste.translogger import TransLogger

app = Flask(__name__)
app.debug = True

def save_classifier(classifier, modelPath):
    f = open(modelPath + '//classifier.pickle', 'wb')
    pickle.dump(classifier, f, -1)
    f.close()

def load_classifier(modelPath):
    f = open(modelPath + '//classifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()
    return classifier


def construct_model(copusPath, modelPath):
    mr = CategorizedPlaintextCorpusReader(copusPath, r'(?!\.).*\.txt',
                                           cat_pattern=r'*/.*', encoding='iso-8859-1')
    stop = stopwords.words('french')
    documents = [([w for w in mr.words(i) if w.lower() not in stop and w.lower() not in string.punctuation],
                   i.split('/')[0]) for i in mr.fileids()]
    word_features = FreqDist(chain(*[i for i, j in documents]))
    word_features = list(word_features.keys())
    numtrain = int(len(documents) * 100 / 100)
    train_set = [({i:(i in tokens) for i in word_features}, tag) for tokens, tag in documents[:numtrain]]
    """test_set = [({i:(i in tokens) for i in word_features}, tag) for tokens, tag  in documents[numtrain:]]"""
    classifier = nbc.train(train_set)
    mrtest = CategorizedPlaintextCorpusReader(os.path.abspath("corpus_test"), r'(?!\.).*\.txt', cat_pattern=r'*/.*', encoding='iso-8859-1')
    documentsTest = [([w for w in mrtest.words(i) if w.lower() not in stop and w.lower() 
                   not in string.punctuation],
                   i.split('/')[0]) for i in mrtest.fileids()]
    word_features_test = FreqDist(chain(*[i for i, j in documentsTest]))
    word_features_test = list(word_features_test.keys())
    numtrain_test = int(len(documentsTest) * 100 / 100)
    test_set = [({i:(i in tokens) for i in word_features_test}, tag) for tokens, tag  in documentsTest[:numtrain_test]]
    save_classifier(classifier, modelPath)


def classify(words, modelPath):
    category=""
    feats = dict([(word, True) for word in words])
    classifier = load_classifier(modelPath)
    classifier.classify(feats)
    dist = classifier.prob_classify(feats)
    for label in dist.samples():
        print("%s: %f" % (label, dist.prob(label)))
    if dist.prob(dist.max())>0.5:
        category=dist.max()
    return category



app = Flask(__name__)

@app.route('/NLP/api/v1.0/classfierModel', methods=['GET'])
def trainModel():  
    construct_model(os.path.abspath("corpus"), os.path.abspath("model"))
    return send_from_directory(directory=os.path.abspath("model"), filename="classifier.pickle", as_attachment=True)


@app.route('/NLP/api/v1.0/categories', methods=['POST'])
def getCategorie():
    if not request.json or not 'text' in request.json:
        abort(400)
    modelPath = os.path.abspath("model")
    print(request.json['text'])  
    return jsonify({'categorie': classify(request.json['text'].split(),modelPath)}), 200


@app.route('/NLP/api/v1.0/posTagger', methods=['POST'])
def posTagText():
    dependenciesPath = os.path.abspath("dependencies")
    if not request.json or not 'text' in request.json:
        abort(400)  
    print(request.json['text'])
    return jsonify({'entities': stanfordTag(dependenciesPath+"/french.tagger",
                                            dependenciesPath+"/stanford-postagger.jar",
                                            request.json['text'],encoding='utf8')}), 200

def stanfordTag(modelPath,stanfordJarPath,text,encoding):
    print(stanfordJarPath) 
    if not bool(re.search("java", os.getenv("JAVA_HOME"))):
        java_path=os.getenv("JAVA_HOME")+"bin/java"
        os.environ['JAVA_HOME'] = java_path
        print(java_path)
        nltk.internals.config_java(java_path)
    entities = []
    stemmer = SnowballStemmer("french")
    st = POSTagger(modelPath,stanfordJarPath,encoding) 
    print(text.split())
    tags=st.tag(text.split())
    print(tags)
    for tag in tags:
        print(tag)        
        entity = {
        'token': tag[0],
        'pos': tag[1],
        'stemm' : stemmer.stem(tag[0])       
        }
        entities.append(entity)
    return entities

    
def run_server():
    # Enable WSGI access logging via Paste
    app_logged = TransLogger(app)

    # Mount the WSGI callable object (app) on the root directory
    cherrypy.tree.graft(app_logged, '/')

    # Set the configuration of the web server
    cherrypy.config.update({
        'engine.autoreload_on': True,
        'log.screen': True,
        'server.socket_port': 5000,
        'server.socket_host': '0.0.0.0'
    })

    # Start the CherryPy WSGI web server
    cherrypy.engine.start()
    cherrypy.engine.block()

if __name__ == "__main__":
    run_server()


    
