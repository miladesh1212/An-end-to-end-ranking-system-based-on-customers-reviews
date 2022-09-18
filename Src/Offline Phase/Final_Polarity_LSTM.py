# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 09:47:44 2019

@author: Milad
"""

import numpy as np 
#import spacy
##import pdb;pdb.set_trace()
#nlp = spacy.load('en_core_web_sm')
import xml.etree.ElementTree as ET
from lxml import etree
from nltk import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from keras.layers import LSTM
from keras.layers import Bidirectional
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Binarizer
import string
from sklearn.metrics import precision_recall_fscore_support
from keras.utils import to_categorical
import tensorflow as tf
from sklearn.metrics import accuracy_score

## Functions 

separator = ['and', ';', ',', '...', '(', 'so', 'yet', 'unless', ':', 'at', 'but', '-', 'with', 'though', 'although', '--', 'while', 'about', 'for', 'including', 'especially', 'in', '..', 'to', 'until',' ;', ' ,', '...',]

def getSentences(file):
	tree = ET.parse(file, etree.XMLParser(recover=True, encoding="utf-8"))
	root = tree.getroot()
	s = []
	p = []
	for review in root.findall('Review'):
		for sentences in review.findall('sentences'):
			for sentence in sentences.findall('sentence'):
				text = sentence.find('text').text
				s.append(text)
				polarity = []
				for opinions in sentence.findall('Opinions'):
					for opinion in opinions.findall('Opinion'):
						elem = [opinion.get('category'), opinion.get('polarity'), opinion.get('target'), opinion.get('from'), opinion.get('to')]
						polarity.append(elem)
				p.append(polarity)
				
	return s, p


def getSentences2(file):
  tree = ET.parse(file, etree.XMLParser(recover=True, encoding="utf-8"))
  root = tree.getroot()
  Sentences = []
  Stars = []
  Names = []
  for Item in root.findall('item'):
    Name = []
    Star = []
    for hotel_name in Item.findall('hotel_name'):
        text = hotel_name.find('value').text
        Name.append(text)
       
    for stars in Item.findall('stars'):         
        star = stars.text
        Star.append(star)
        
    for sentence in Item.findall('content'):
        text = sentence.find('value').text
        text= text.split(".")
        for texts in text:
            texts = texts+'.'
            Sentences.append(texts)
      
    Stars.append(Star)
    Names.append(Name)    
  return Sentences, Stars, Names

def getSentences1(file):
	tree = ET.parse(file, etree.XMLParser(recover=True, encoding="utf-8"))
	root = tree.getroot()
	s = []
	p = []
	for review in root.findall('Review'):
		for sentences in review.findall('sentences'):
			for sentence in sentences.findall('sentence'):
				for opinions in sentence.findall('Opinions'):
					for opinion in opinions.findall('Opinion'):
						if opinion.get('target') != 'NULL': s.append(opinion.get('target')); p.append(opinion.get('category'))
			                 
	return s, p
 

def number_of_null_targets(target):
	contor = 0
	for t in target:
		if t == 'NULL':
			contor = contor + 1
	return contor
	
def getSentenceSeparators(words):
	sep = []
	for i, w in enumerate(words):
		if w in separator:
			sep.append(i)
	return sep
	

def getFrequenceFeatures(data):	
	vectorizer = CountVectorizer(analyzer='word', lowercase=True ,stop_words='english',)
	features = vectorizer.fit_transform(data) #Unigram features
	return features, vectorizer
	
def getPresenceFeatures(data):
	vectorizer = CountVectorizer(analyzer='char', lowercase=True ,stop_words='english',)
	features2 = vectorizer.fit_transform(data)#.toarray() #Unigram features
	
	bin = Binarizer()
	presenceFeatures = bin.fit_transform(features2)
	return presenceFeatures, vectorizer
	
def getBigramFeatures(data):
	vectorizer = CountVectorizer(analyzer='word', lowercase=True , ngram_range=(1,2),)
	features = vectorizer.fit_transform(data)
	
	bin = Binarizer()
	bgFt = bin.fit_transform(features)

	
	return bgFt, vectorizer
	
def getTrigramFeatures(data):
	vectorizer = CountVectorizer(analyzer='word', lowercase=False, ngram_range=(1,3),)
	features = vectorizer.fit_transform(data)
	
	bin = Binarizer()
	tgFt = bin.fit_transform(features)
	return tgFt, vectorizer


negation_words = ["no", "not", "none", "nobody", "nothing", "neither", "nowhere", "never", "hardly", "scarcely", "barely", "doesn't", "isn't", "wasn't", "wouldn't", "couldn't", "won't", "can't", "don't"]
def mark_not(sentences):
	changed_sentences = []; Num = -1
	for sentence in sentences:
		words = word_tokenize(sentence); Num +=1; 
		ok = 0
		for i in range(len(words)):
			if ok == 1 and words[i] not in negation_words and words[i] not in string.punctuation:
				words[i] = words[i] + "_NOT"
			if ok == 1 and words[i] in string.punctuation:
				ok = 0
			if words[i] in negation_words:
				if ok == 1:
					ok = 0
				else:
					ok = 1
		changed_sentence = " ".join(words)
		changed_sentences.append(changed_sentence)
    
	return changed_sentences
	
def Delite(sentences, Aspects, stops):
    word_list = word_tokenize(sentences)
    sentence = []
    for word in word_list:
        if word not in Aspects:
            sentence.append(word)
    sentences = " ".join(sentence)    
    return sentences

def Positives(sentences, Words):
    word_list = word_tokenize(sentences)
    sentence = []
    for word in word_list:
        if word in Words:
            sentence.append('Positive')
        elif word not in Words:
            sentence.append(word)
            
    sentences = " ".join(sentence)    
    return sentences

def Negatives(sentences, Words):
    word_list = word_tokenize(sentences)
    sentence = []
    for word in word_list:
        if word in Words:
            sentence.append('Negative')
        elif word not in Words:
            sentence.append(word)
            
    sentences = " ".join(sentence)    
    return sentences

def Unique(List):
    output = []
    for x in List:
        if x not in output:
            output.append(x)
        
    return output    
    
    
    
def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append( ' ' ) #different object reference each time
    return list_of_objects

def Split_Sentence_to_subs(sentences, Targets, separators):
    words = sentences.split(' ')
    tar = init_list_of_objects(len(Targets)*3)
    temp1 = 0
    temp2 = 0
    temp3 = 0
    temp4 = 0
    for i, w in enumerate(words):
        if w in Targets:
            temp1 +=1
            temp4 +=1
        elif w in separators:
            temp2 +=1
            
        if temp1<=1 and temp2==0:
            tem = tar[temp3]
            tar[temp3] = ' '+ tem[1:] +' ' + words[i]
        elif temp2==1 and temp3<(3*len(Targets)-1):
            temp3 +=1
            temp1 = 0
            temp2 = 0
        elif temp4==len(Targets) and temp2==1:
            temp1 = 0
            temp2 = 0  
    
    
    Out = []
    for j in range(len(Targets)):
        for k in range(len(tar)):
            if Targets[j] in tar[k]:
                Out.append(tar[k])
#    import pdb; pdb.set_trace()   
    return Out  


     
########### Main code

## Load data	
train_file = 'data\\resttrain.xml'
test_file = 'data\\resttest.xml.gold'
text_file = open("pos_words.txt", "r")
pos_words = text_file.read().split('\n')
text_file = open("neg_words.txt", "r")
neg_words = text_file.read().split('\n')

# train data
train_sentences, train_data = getSentences(train_file)

# test data
test_sentences, test_data = getSentences(test_file)

## Load Crawled data for costruct a big word Embeding model which is proper for online phase   
All_Sentences, Stars2, Names2 = getSentences2("crawled_data\\tripadvisor1.xml")

unique_not_list, unique_not_list1 = getSentences1("data\\restaurants\\train.xml")
Aspects = []
Categories = []  
for i in range(len(unique_not_list)): 
    # check if exists in unique_list or not 
    if unique_not_list[i] not in Aspects: 
        Aspects.append(unique_not_list[i]) 
        Categories.append(unique_not_list1[i]) 
        

## form train dataset
train_sentences_1 = []
train_data_1 = []
for i in range(len(train_data)):
    temp = train_data[i]
    if len(temp)==1 and temp[0][2]!='NULL':
       temp3 = train_sentences[i]
       train_sentences_1.append(temp3.replace(temp[0][2], temp[0][0]))
       train_data_1.append(temp[0][1])

    elif len(temp)>1:
        Targets = []
        temp3 = train_sentences[i]
        for j in range(len(temp)):
            if temp[j][2]!='NULL':
                temp1 = temp[j]
#                train_data_1.append(temp1[1])
                temp3 = temp3.replace(temp1[2], temp1[0])
                Targets.append(temp1[0])
  
                temp2 = Split_Sentence_to_subs(temp3, Targets, separator)

                if len(temp2)==len(Targets):                    
                    for k in range(len(temp2)):
                        train_sentences_1.append(temp2[k])
                        try:
                            train_data_1.append(temp[k][1])
                        except:
                            import pdb; pdb.set_trace();
                        
                elif len(temp2)>len(Targets):
                    temp4 = Unique(temp2)
                    temp4 = temp4[0:len(Targets)-1]
                    for k in range(len(temp4)):
                        train_sentences_1.append(temp4[k])
                        try:
                            train_data_1.append(temp[k][1])
                        except:
                            import pdb; pdb.set_trace();
                elif len(temp2)<len(Targets):
                    temp4 = temp2[0:len(Targets)-1]
                    for k in range(len(temp4)):
                        train_sentences_1.append(temp4[k])
                        try:
                            train_data_1.append(temp[k][1])
                        except:
                            import pdb; pdb.set_trace();
            

## form test dataset
test_sentences_1 = []
test_data_1 = []
for i in range(len(test_data)):
    temp = test_data[i]
    if len(temp)==1 and temp[0][2]!='NULL':
       temp3 = test_sentences[i]
       test_sentences_1.append(temp3.replace(temp[0][2], temp[0][0]))
       test_data_1.append(temp[0][1])

    elif len(temp)>1:
        Targets = []
        temp3 = test_sentences[i]
        for j in range(len(temp)):
            if temp[j][2]!='NULL':
                temp1 = temp[j]
#                train_data_1.append(temp1[1])
                temp3 = temp3.replace(temp1[2], temp1[0])
                Targets.append(temp1[0])
  
                temp2 = Split_Sentence_to_subs(temp3, Targets, separator)

                if len(temp2)==len(Targets):                    
                    for k in range(len(temp2)):
                        test_sentences_1.append(temp2[k])
                        try:
                            test_data_1.append(temp[k][1])
                        except:
                            import pdb; pdb.set_trace();
                        
                elif len(temp2)>len(Targets):
                    temp4 = Unique(temp2)
                    temp4 = temp4[0:len(Targets)-1]
                    for k in range(len(temp4)):
                        test_sentences_1.append(temp4[k])
                        try:
                            test_data_1.append(temp[k][1])
                        except:
                            import pdb; pdb.set_trace();
                elif len(temp2)<len(Targets):
                    temp4 = temp2[0:len(Targets)-1]
                    for k in range(len(temp4)):
                        test_sentences_1.append(temp4[k])
                        try:
                            test_data_1.append(temp[k][1])
                        except:
                            import pdb; pdb.set_trace();

# Detect Positive words in All sentences
All_Sentences_N1 = []
for i in range(len(All_Sentences)):
    All_Sentences_N1.append(Positives(All_Sentences[i], pos_words))    
    
# Detect Negative words in All sentences
All_Sentences_N2 = []        
for i in range(len(All_Sentences_N1)):
    All_Sentences_N2.append(Negatives(All_Sentences_N1[i], neg_words))  

    
# Detect Positive words in train sentences
train_sentences_prelucrate_N1 = []        
for i in range(len(train_sentences_1)):
    train_sentences_prelucrate_N1.append(Positives(train_sentences_1[i], pos_words))

# Detect Positive words in test sentences
test_sentences_prelucrate_N1 = []
for i in range(len(test_sentences_1)):
    test_sentences_prelucrate_N1.append(Positives(test_sentences_1[i], pos_words))    
    
# Detect Negative words in train sentences
train_sentences_prelucrate_N2 = []        
for i in range(len(train_sentences_prelucrate_N1)):
    train_sentences_prelucrate_N2.append(Negatives(train_sentences_prelucrate_N1[i], neg_words))

# Detect Negative words in test sentences
test_sentences_prelucrate_N2 = []
for i in range(len(test_sentences_prelucrate_N1)):
    test_sentences_prelucrate_N2.append(Negatives(test_sentences_prelucrate_N1[i], neg_words))      

#Add Not to words after Negation words
#train_sentences_prelucrate_N2 = mark_not(train_sentences_prelucrate_N2) 
#test_sentences_prelucrate_N2 = mark_not(test_sentences_prelucrate_N2)
#All_Sentences_N2 = mark_not(All_Sentences_N2)

#Creating the train Labels
train_labels = []
for item in train_data_1:
    if item == 'positive':
        train_labels.append(2)
    if item == 'negative':
        train_labels.append(0)       
    if item == 'neutral':
        train_labels.append(1)      

train_labels = to_categorical(train_labels)   

#Handling the testing
test_labels = []
for item in test_data_1:
    if item == 'positive':
        test_labels.append(2)
    if item == 'negative':
        test_labels.append(0)     
    if item == 'neutral':
        test_labels.append(1)
      

test_labels = to_categorical(test_labels)       

# Merge Crawled Data and Train Data    
for review in train_sentences_prelucrate_N2:
  All_Sentences_N2.append(review)   
    
#import pdb; pdb.set_trace()

#Vectorizing data
vectorizer = CountVectorizer(analyzer='word', lowercase=True, stop_words='english', ngram_range=(1,1))
vectorizer.fit(All_Sentences_N2)

x_train = vectorizer.transform(train_sentences_prelucrate_N2)
x_test = vectorizer.transform(test_sentences_prelucrate_N2)
All = vectorizer.transform(All_Sentences_N2)

input_dim = x_train.shape[1]

tokenizer = Tokenizer(num_words = 20000)
tokenizer.fit_on_texts(All_Sentences_N2)

x_train = tokenizer.texts_to_sequences(train_sentences_prelucrate_N2)
x_test = tokenizer.texts_to_sequences(test_sentences_prelucrate_N2)
vocab_size = len(tokenizer.word_index) + 1

maxlen = 100
x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
x_test = pad_sequences(x_test, padding = 'post', maxlen=maxlen)

#Pretrained Word Embeddings
def create_embedding_matrix(filepath, word_index, embedding_dim):
	vocab_size = len(word_index) + 1
	embedding_matrix = np.zeros((vocab_size, embedding_dim))
	
	with open(filepath, encoding='utf-8') as f:
		for line in f:
			word, *vector = line.split()
			if word in word_index:
				idx = word_index[word]
				embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]

	return embedding_matrix

embedding_dim = 200
embedding_matrix = create_embedding_matrix('data\\glove.6B.200d.txt', tokenizer.word_index, embedding_dim)

nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis = 1))
#print(nonzero_elements / vocab_size)

def f1(y_true, y_pred):
    TP = tf.count_nonzero(y_pred * y_true)
    TN = tf.count_nonzero((y_pred - 1) * (y_true - 1))
    FP = tf.count_nonzero(y_pred * (y_true - 1))
    FN = tf.count_nonzero((y_pred - 1) * y_true)
    precision1 = TP / (TP + FP)
    recall1 = TP / (TP + FN)
    precision2 = TN/ (TN+FP)
    recall2 = TN / (TN + FP)
    f1_1 = 2 * precision1 * recall1 / (precision1 + recall1)
    f1_2 = 2 * precision2 * recall2 / (precision2 + recall2)
    result = (f1_1 + f1_2) /2
    return result
#
#def f2_score(y_true, y_pred):
#    y_true = tf.cast(y_true, "int32")
#    y_pred = tf.cast(tf.round(y_pred), "int32") # implicit 0.5 threshold via tf.round
#    y_correct = y_true * y_pred
#    sum_true = tf.reduce_sum(y_true, axis=1)
#    sum_pred = tf.reduce_sum(y_pred, axis=1)
#    sum_correct = tf.reduce_sum(y_correct, axis=1)
#    precision = sum_correct / sum_pred
#    recall = sum_correct / sum_true
#    f_score = 5 * precision * recall / (4 * precision + recall)
#    f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
#    return tf.reduce_mean(f_score)
#
#def getPredictions(x_train, x_test, train, test):
#    embedding_dim = 200
#    embedding_matrix = create_embedding_matrix('data\\glove.6B.200d.txt', tokenizer.word_index, embedding_dim)
#    model = Sequential()
#    model.add(layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length = maxlen, trainable = True))
#    model.add(Bidirectional(LSTM(200, dropout=0.2, recurrent_dropout=0.2)))
#    model.add(layers.Dense(100, activation='relu')); 
#    model.add(layers.Dense(20, activation='relu'));
#    model.add(layers.Dense(3, activation='softmax'))
#    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f2_score, 'accuracy'])
#    model.fit(x_train, train, epochs = 20, verbose = False, validation_data = (x_test, test), batch_size = 10)
#    model.evaluate(x_train, train, verbose = False)
#    predictions = model.predict(x_test)
#    return predictions, model


#print('')
#print('')
#print('================================================================================================')
#print('LSTM+WordEmbeding+MonogramFeatures')
#print("Getting Predictions1")
#predictions1, model = getPredictions(x_train, x_test, train_labels, test_labels)
#result1 = precision_recall_fscore_support(test_labels, np.round(predictions1))
#
#print("F1: ", result1[2][0])
#print("Accuracy: ", accuracy_score(test_labels, np.round(predictions1)))
#
#
#
##Vectorizing data
#vectorizer = CountVectorizer(analyzer='word', lowercase=True, stop_words='english', ngram_range=(1,2))
#vectorizer.fit(All_Sentences_N2)
#
#x_train = vectorizer.transform(train_sentences_prelucrate_N2)
#x_test = vectorizer.transform(test_sentences_prelucrate_N2)
#All = vectorizer.transform(All_Sentences_N2)

input_dim = x_train.shape[1]

tokenizer = Tokenizer(num_words = 20000)
tokenizer.fit_on_texts(All_Sentences_N2)

x_train = tokenizer.texts_to_sequences(train_sentences_prelucrate_N2)
x_test = tokenizer.texts_to_sequences(test_sentences_prelucrate_N2)
vocab_size = len(tokenizer.word_index) + 1
import pdb; pdb.set_trace()
maxlen = 100
x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
x_test = pad_sequences(x_test, padding = 'post', maxlen=maxlen)

#Pretrained Word Embeddings
def create_embedding_matrix(filepath, word_index, embedding_dim):
	vocab_size = len(word_index) + 1
	embedding_matrix = np.zeros((vocab_size, embedding_dim))
	
	with open(filepath, encoding='utf-8') as f:
		for line in f:
			word, *vector = line.split()
			if word in word_index:
				idx = word_index[word]
				embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]

	return embedding_matrix

embedding_dim = 200
embedding_matrix = create_embedding_matrix('data\\glove.6B.200d.txt', tokenizer.word_index, embedding_dim)

nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis = 1))


print('')
print('')
print('================================================================================================')
print('LSTM+WordEmbeding+BigramFeatures')
print("Getting Predictions1")
predictions1, model = getPredictions(x_train, x_test, train_labels, test_labels)
result1 = precision_recall_fscore_support(test_labels, np.round(predictions1))

print("F1: ", result1[2][0])
print("Accuracy: ", accuracy_score(test_labels, np.round(predictions1)))




#Vectorizing data
vectorizer = CountVectorizer(analyzer='word', lowercase=True, stop_words='english', ngram_range=(1,3))
vectorizer.fit(All_Sentences_N2)

x_train = vectorizer.transform(train_sentences_prelucrate_N2)
x_test = vectorizer.transform(test_sentences_prelucrate_N2)
All = vectorizer.transform(All_Sentences_N2)

input_dim = x_train.shape[1]

tokenizer = Tokenizer(num_words = 20000)
tokenizer.fit_on_texts(All_Sentences_N2)

x_train = tokenizer.texts_to_sequences(train_sentences_prelucrate_N2)
x_test = tokenizer.texts_to_sequences(test_sentences_prelucrate_N2)
vocab_size = len(tokenizer.word_index) + 1

maxlen = 100
x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
x_test = pad_sequences(x_test, padding = 'post', maxlen=maxlen)

#Pretrained Word Embeddings
def create_embedding_matrix(filepath, word_index, embedding_dim):
	vocab_size = len(word_index) + 1
	embedding_matrix = np.zeros((vocab_size, embedding_dim))
	
	with open(filepath, encoding='utf-8') as f:
		for line in f:
			word, *vector = line.split()
			if word in word_index:
				idx = word_index[word]
				embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]

	return embedding_matrix

embedding_dim = 200
embedding_matrix = create_embedding_matrix('data\\glove.6B.200d.txt', tokenizer.word_index, embedding_dim)

nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis = 1))


print('')
print('')
print('================================================================================================')
print('LSTM+WordEmbeding+TrigramFeatures')
print("Getting Predictions1")
predictions1, model = getPredictions(x_train, x_test, train_labels, test_labels)
result1 = precision_recall_fscore_support(test_labels, np.round(predictions1))

print("F1: ", result1[2][0])
print("Accuracy: ", accuracy_score(test_labels, np.round(predictions1)))
