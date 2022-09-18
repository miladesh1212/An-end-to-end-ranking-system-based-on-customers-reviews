# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 09:30:15 2019

@author: Milad
"""

import spacy
import numpy as np
nlp = spacy.load('en_core_web_sm')
import xml.etree.ElementTree as ET
from lxml import etree
from nltk import word_tokenize
from sklearn.preprocessing import Binarizer
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
import scipy.sparse as sp
from sklearn.neighbors.nearest_centroid import NearestCentroid
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
train_sentences_prelucrate_N2 = mark_not(train_sentences_prelucrate_N2) 
test_sentences_prelucrate_N2 = mark_not(test_sentences_prelucrate_N2)
All_Sentences_N2 = mark_not(All_Sentences_N2)

#Creating the train Labels
train_labels = []
for item in train_data_1:
    if item == 'positive':
        train_labels.append(2)
    if item == 'negative':
        train_labels.append(0)       
    if item == 'neutral':
        train_labels.append(1)      
   

#Handling the testing
test_labels = []
for item in test_data_1:
    if item == 'positive':
        test_labels.append(2)
    if item == 'negative':
        test_labels.append(0)     
    if item == 'neutral':
        test_labels.append(1)
      
     

# Merge Crawled Data and Train Data    
for review in train_sentences_prelucrate_N2:
  All_Sentences_N2.append(review)     
 
# Features    
dd, vectorizer0 = getFrequenceFeatures(All_Sentences_N2)
ff, vectorizer1 = getBigramFeatures(All_Sentences_N2)
vv, vectorizer2 = getTrigramFeatures(All_Sentences_N2)

features0 = vectorizer0.transform(train_sentences_prelucrate_N2)
features1 = vectorizer1.transform(train_sentences_prelucrate_N2)
features2 = vectorizer2.transform(train_sentences_prelucrate_N2)
features = sp.hstack((features0,features1,features2))

test_features0 = vectorizer0.transform(test_sentences_prelucrate_N2)
test_features1 = vectorizer1.transform(test_sentences_prelucrate_N2)
test_features2 = vectorizer2.transform(test_sentences_prelucrate_N2)
test_features = sp.hstack((test_features0,test_features1,test_features2))


# Results
print('')
print('')
print('================================================================================================')
print('KNN+FrequenceFeatures')
classifier = NearestCentroid()
model = classifier.fit(features0, train_labels)
prediction = model.predict(test_features0)
result = precision_recall_fscore_support(test_labels, prediction.tolist()) 
print("F1: ", result[2][0])
print("Accuracy: ", accuracy_score(test_labels, prediction.tolist()))


print('')
print('')
print('================================================================================================')
print('KNN+BigramFeatures')
classifier = NearestCentroid()
model = classifier.fit(features1, train_labels)
prediction = model.predict(test_features1)
result = precision_recall_fscore_support(test_labels, prediction.tolist()) 
print("F1: ", result[2][0])
print("Accuracy: ", accuracy_score(test_labels, prediction.tolist()))


print('')
print('')
print('================================================================================================')
print('KNN+TrigramFeatures')
classifier = NearestCentroid()
model = classifier.fit(features2, train_labels)
prediction = model.predict(test_features2)
result = precision_recall_fscore_support(test_labels, prediction.tolist()) 
print("F1: ", result[2][0])
print("Accuracy: ", accuracy_score(test_labels, prediction.tolist()))


print('')
print('')
print('================================================================================================')
print('KNN+FrequenceFeatures+BigramFeatures+TrigramFeatures')
classifier = NearestCentroid()
model = classifier.fit(features, train_labels)
prediction = model.predict(test_features)
result = precision_recall_fscore_support(test_labels, prediction.tolist()) 
print("F1: ", result[2][0])
print("Accuracy: ", accuracy_score(test_labels, prediction.tolist()))






#allfeatures, vectorizer1 = getFrequenceFeatures(All_Sentences_N2)
#joblib.dump(vectorizer, 'Polarity_vectorizer.pkl') 
#features = vectorizer.transform(train_sentences_prelucrate_N2)
# Output a pickle file for the model
#joblib.dump(classifier, 'Polarity_model.pkl') 