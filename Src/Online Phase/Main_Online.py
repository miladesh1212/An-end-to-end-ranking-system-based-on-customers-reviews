# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 16:54:36 2019

@author: Milad
"""

## Libraries
import pandas as pd
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk import pos_tag
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import Binarizer
plt.style.use('ggplot')
import numpy as np
import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy, os
from lxml import etree
import string
import dawid_skene
from bw_method_weight_scipy import calc_weight
import scipy.sparse as sp


## Functions 
separator = ['and', ';', ',', '...', '(', 'so', 'yet', 'unless', ':', 'at', 'but', '-', 'with', 'though', 'although', '--', 'while', 'about', 'for', 'including', 'especially', 'in', '..', 'to', 'until',' ;', ' ,', '...',]

def getSentences(file):
  tree = ET.parse(file, etree.XMLParser(recover=True, encoding="utf-8"))
  root = tree.getroot()
  Sentences = []
  Star = []
  date = []
  Star = []
  Name = []
  for Item in root.findall('item'):
      
    for hotel_name in Item.findall('hotel_name'):
        text = hotel_name.text
        Name.append(text)

    for dates in Item.findall('date'):
        text = dates.text
        date.append(text)
        
    for stars in Item.findall('hotel_star'):         
        star = stars.text
        Star.append(star)
    Sentence = []    
    for sentence in Item.findall('content'):
        text = sentence.find('value').text
        text= text.split(".")
        for texts in text:
            texts = texts+'.'
            Sentence.append(texts)
    Sentences.append(  Sentence)   
  return Sentences, Star, Name, date

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


def getPredictions_Sub_1_2(Sentence, Aspects, Categories):
    Sentence = Sentence.replace('.', ' .')
    Sentence = Sentence.replace(',', ' , ')
    Sentence = Sentence.replace(';', ' ;')
    Sentence = Sentence.replace('"', ' ')
    try:
        predictions = []
        for i in range(len(Aspects)):
            if Aspects[i] in Sentence.split():
                predictions.append(Categories[i])
                Sentence = Sentence.replace(Aspects[i], Categories[i])
    except:
        import pdb; pdb.set_trace()
            
    return predictions, Sentence

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
    tar = init_list_of_objects(len(Targets)*5)
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
        elif temp2==1 and temp3<(5*len(Targets)-1):
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

def Get_Polarity_predict(model_Polarity, features):
    Predict = model_Polarity.predict(features)
    Predict = np.transpose(Predict)
    return Predict


################# Read crawled data and psitive and negatives
    
Sentences, Stars, Names, Dates = getSentences("crawled_data\\tripadvisor3.xml")
text_file = open("pos_words.txt", "r")
pos_words = text_file.read().split('\n')
text_file = open("neg_words.txt", "r")
neg_words = text_file.read().split('\n')

########## Read Aspects and categories form semeval data

unique_not_list, unique_not_list1 = getSentences1("data\\restaurants\\train.xml")
Aspects = []
Categories = []  
for i in range(len(unique_not_list)): 
    # check if exists in unique_list or not 
    if unique_not_list[i] not in Aspects: 
        Aspects.append(unique_not_list[i].lower()) 
        Categories.append(unique_not_list1[i]) 

        
#import pdb; pdb.set_trace()         
###### Subtask 1 and 2 ########
"""
First Change Data shape for  are equal to requirment
"""
Current_Star = '5';
Subtask2_Sentences = []
Subtask2_Dates = []
Subtask2_Names = []
for i in range(len(Stars)):
    if Stars[i] == Current_Star:
        Subtask2_Dates.append(Dates[i])
        Subtask2_Names.append(Names[i])
        Sents = []
        for sent in Sentences[i]:
            Sents.append(sent)
        Subtask2_Sentences.append(Sents)

Subtask2_Sentence = [] 
Information = [] 
temp = 0;      
for review in Subtask2_Sentences:
    temp +=len(review)
    Information.append(temp)
    for item in review:
        Subtask2_Sentence.append(item.lower()) 
        

predicted_aspects = []
New_Subtask2_Sentence = []
nr = len(Subtask2_Sentence)
for i in range(nr):
  predicted_aspect, Sentence_Prim = getPredictions_Sub_1_2(Subtask2_Sentence[i], Aspects, Categories) 
  predicted_aspects.append(predicted_aspect)
  New_Subtask2_Sentence.append(Sentence_Prim)
  
  
  
################## Subtask 3 ##############


## form test dataset, split a whol sentense to sunsentence wich contain aspects
New_Subtask2_Sentence_1 = []
New_predicted_aspects = []
for i in range(len(predicted_aspects)):
    temp = predicted_aspects[i]
    if len(temp)<=1:        
        New_Subtask2_Sentence_1.append(New_Subtask2_Sentence[i])
        New_predicted_aspects.append(temp)

    elif len(temp)>1:        
        temp3 = New_Subtask2_Sentence[i]
        temp2 = Split_Sentence_to_subs(temp3, temp, separator)
        
        if len(temp2)==len(temp):
            New_Subtask2_Sentence_1.append(temp2)
            New_predicted_aspects.append(temp)
                                    
        elif len(temp2)>len(temp):
            target = []
            for k in range(len(temp)):
                for q in range(len(temp2)):
                    if temp[k] in temp2[q]:
                        target.append(temp2[q])
                       
            New_Subtask2_Sentence_1.append(target[0:len(temp)])
            New_predicted_aspects.append(temp)

        elif len(temp2)<len(temp):
            if len(temp2)==1:
                temp4 = [temp[0]]
                New_Subtask2_Sentence_1.append(temp2[0])
                New_predicted_aspects.append(temp4)                
            elif len(temp2)>1:
                temp4 = temp[0:len(temp2)]
                New_Subtask2_Sentence_1.append(temp2)
                New_predicted_aspects.append(temp4)

# Preprocessing
New_Subtask2_Sentence_2 = []
for i in range(len(New_predicted_aspects)):
    if len(New_predicted_aspects[i])==0:
        New_Subtask2_Sentence_2.append([])
    elif len(New_predicted_aspects[i])==1:
        # Detect Positive words in sentences
        Temp = Positives(New_Subtask2_Sentence_1[i], pos_words)
        # Detect Negative words in sentences
        Temp = Negatives(Temp, neg_words)
        #Add Not to words after Negation words
        Temp = mark_not([Temp]) 
        New_Subtask2_Sentence_2.append(Temp[0])
    elif len(New_predicted_aspects[i])>1:
        temp1 = []
        for j in range(len(New_predicted_aspects[i])):
            temp2 = New_Subtask2_Sentence_1[i][j]
            # Detect Positive words in sentences
            temp2 = Positives(temp2, pos_words)
            # Detect Negative words in sentences
            temp2 = Negatives(temp2, neg_words)
            #Add Not to words after Negation words
            temp2 = mark_not([temp2]) 
            temp1.append(temp2[0])            
        
        New_Subtask2_Sentence_2.append(temp1)
    

#import pdb; pdb.set_trace()
# Import Saved models in Offline Phase        
vectorizer0 = joblib.load('Polarity_vectorizer0.pkl')
vectorizer1 = joblib.load('Polarity_vectorizer1.pkl')
vectorizer2 = joblib.load('Polarity_vectorizer2.pkl')

# Load the pickle file
model_Polarity = joblib.load('Polarity_model.pkl')

## Getting Predictions Polarity
print('')
print('===================================================================================')
print("Getting Predictions Polarity")

prediction = []
#range(len(New_predicted_aspects))
for i in range(len(New_predicted_aspects)):
    if len(New_predicted_aspects[i])==0:
        prediction.append([])
    elif len(New_predicted_aspects[i])==1:
        features0 = vectorizer0.transform([New_Subtask2_Sentence_2[i]])
        features1 = vectorizer1.transform([New_Subtask2_Sentence_2[i]])
        features2 = vectorizer2.transform([New_Subtask2_Sentence_2[i]])
        features = sp.hstack((features0,features1,features2))
        temp = Get_Polarity_predict(model_Polarity, features)
        prediction.append(temp)

    elif len(New_predicted_aspects[i])>1:
        temp1 = []
        for j in range(len(New_predicted_aspects[i])):
            features0 = vectorizer0.transform([New_Subtask2_Sentence_2[i][j]])
            features1 = vectorizer1.transform([New_Subtask2_Sentence_2[i][j]])
            features2 = vectorizer2.transform([New_Subtask2_Sentence_2[i][j]])
            features = sp.hstack((features0,features1,features2))
            temp = Get_Polarity_predict(model_Polarity, features)            
            temp1.append(temp)
        prediction.append(temp1)
    

#import pdb; pdb.set_trace()
###### Reshape results ###################

Aspects = []
Poarities = []
for i in range(len(Information)):
    if i == 0:
        temp = []
        temp1 = []
        for q in range(0, Information[i]):
            terms = New_predicted_aspects[q]
            if len (terms)==0:
                temp1 = []
                temp = []
            elif len (terms)==1: 
                temp1 = terms
                temp = prediction[q].tolist()
            elif len (terms)>1:    
                for k in range(len(terms)): 
                        temp.append(prediction[q][k].tolist())
                        temp1.append(terms[k]) 
        Aspects.append(temp1)
        Poarities.append(temp)
    else:
        temp = []
        temp1 = []
        for q in range(Information[i-1], Information[i]):
            terms = New_predicted_aspects[q]
            if len (terms)==0:
                temp1 = []
                temp = []
            elif len (terms)==1: 
                temp1 = terms
                temp = prediction[q].tolist()
            elif len (terms)>1:    
                for k in range(len(terms)):
                    try:
                        tem = prediction[q][k]
                        temp.append(tem.tolist())
                        temp1.append(terms[k])                         
                    except: 
                        print(q)
                        import pdb; pdb.set_trace()    

        Aspects.append(temp1)
        Poarities.append(temp)   


#import pdb; pdb.set_trace()       
############## get unique hotel names 

U_Name = pd.Series(Subtask2_Names, name='A').unique()
print('Number of reviews:')
print(len(Aspects))
print('Number of Hotels:')
print(len(U_Name))
print(U_Name)
Catego = pd.Categorical(U_Name)

raw_cat = pd.Categorical(Subtask2_Names, categories=U_Name)

Dict_date = {}
Dict_aspect = {}
Dict_polarity = {}
Dict1 = {}
Dict2 = {}
Dict3 = {}
Dict4 = {}
Dict5 = {}
Dict6 = {}
Dict7 = {}
Dict8 = {}
Dict9 = {}
Dict10 = {}
Dict11 = {}
Dict12 = {}

for i in range(len(U_Name)):
    id1 = 0
    id2 = 0
    id3 = 0
    id4 = 0
    id5 = 0
    id6 = 0
    id7 = 0
    id8 = 0
    id9 = 0
    id10 = 0
    id11 = 0
    id12 = 0
    dict1 = {}
    dict2 = {}
    dict3 = {}
    dict4 = {}
    dict5 = {}
    dict6 = {}
    dict7 = {}
    dict8 = {}
    dict9 = {}
    dict10 = {}
    dict11 = {}
    dict12 = {}
    
    Matching = [k for k in range(len(raw_cat)) if U_Name[i] == raw_cat[k]]; 
	
    for l in Matching:
#        import pdb; pdb.set_trace() ; 
        if 'RESTAURANT#GENERAL' in Aspects[l]:
            id1+=1
            temp = {}
            temp1 = str(id1)
            temp2 = int(temp1)
            dict1.update({temp2: [Poarities[l][Aspects[l].index('RESTAURANT#GENERAL')]]})
        else:
            temp = {}
            temp1 = str(id1)
            temp2 = int(temp1)
            dict1.update({temp2: [1]})
        if 'FOOD#QUALITY' in Aspects[l]:
            id2+=1
            temp = {}
            temp1 = str(id2)
            temp2 = int(temp1)
            dict2.update({temp2: [Poarities[l][Aspects[l].index('FOOD#QUALITY')]]})
        else:
            id2+=1
            temp = {}
            temp1 = str(id2)
            temp2 = int(temp1)
            dict2.update({temp2: [1]})
        if 'RESTAURANT#MISCELLANEOUS' in Aspects[l]:
            id3+=1
            temp = {}
            temp1 = str(id3)
            temp2 = int(temp1)
            dict3.update({temp2: [Poarities[l][Aspects[l].index('RESTAURANT#MISCELLANEOUS')]]})
        else:
            id3+=1
            temp = {}
            temp1 = str(id3)
            temp2 = int(temp1)
            dict3.update({temp2: [1]})           
        if 'FOOD#PRICES' in Aspects[l]:
            id4+=1
            temp = {}
            temp1 = str(id4)
            temp2 = int(temp1)
            dict4.update({temp2: [Poarities[l][Aspects[l].index('FOOD#PRICES')]]})
        else:
            id4+=1
            temp = {}
            temp1 = str(id4)
            temp2 = int(temp1)
            dict4.update({temp2: [1]})             
        if 'DRINKS#QUALITY' in Aspects[l]:
            id5+=1
            temp = {}
            temp1 = str(id5)
            temp2 = int(temp1)
            dict5.update({temp2: [Poarities[l][Aspects[l].index('DRINKS#QUALITY')]]})
        else:
            id5+=1
            temp = {}
            temp1 = str(id5)
            temp2 = int(temp1)
            dict5.update({temp2: [1]})                
        if 'LOCATION#GENERAL' in Aspects[l]:
            id6+=1
            temp = {}
            temp1 = str(id6)
            temp2 = int(temp1)
            dict6.update({temp2: [Poarities[l][Aspects[l].index('LOCATION#GENERAL')]]})	
        else:
            id6+=1
            temp = {}
            temp1 = str(id6)
            temp2 = int(temp1)
            dict6.update({temp2: [1]})           
        if 'RESTAURANT#PRICES' in Aspects[l]:
            id7+=1
            temp = {}
            temp1 = str(id7)
            temp2 = int(temp1)
            dict7.update({temp2: [Poarities[l][Aspects[l].index('RESTAURANT#PRICES')]]})	
        else:
            id7+=1
            temp = {}
            temp1 = str(id7)
            temp2 = int(temp1)
            dict7.update({temp2: [1]})			 
        if 'AMBIENCE#GENERAL' in Aspects[l]:
            id8+=1
            temp = {}
            temp1 = str(id8)
            temp2 = int(temp1)
            dict8.update({temp2: [Poarities[l][Aspects[l].index('AMBIENCE#GENERAL')]]})
        else:
            id8+=1
            temp = {}
            temp1 = str(id8)
            temp2 = int(temp1)
            dict8.update({temp2: [1]})
        if 'DRINKS#STYLE_OPTIONS' in Aspects[l]:
            id9+=1
            temp = {}
            temp1 = str(id9)
            temp2 = int(temp1)
            dict9.update({temp2: [Poarities[l][Aspects[l].index('DRINKS#STYLE_OPTIONS')]]})	
        else:
            id9+=1
            temp = {}
            temp1 = str(id9)
            temp2 = int(temp1)
            dict9.update({temp2: [1]})
        if 'SERVICE#GENERAL' in Aspects[l]:
            id10+=1
            temp = {}
            temp1 = str(id10)
            temp2 = int(temp1)
            dict10.update({temp2: [Poarities[l][Aspects[l].index('SERVICE#GENERAL')]]})	
        else:
            id10+=1
            temp = {}
            temp1 = str(id10)
            temp2 = int(temp1)
            dict10.update({temp2: [1]})
        if 'DRINKS#PRICES' in Aspects[l]:
            id11+=1
            temp = {}
            temp1 = str(id11)
            temp2 = int(temp1)
            dict11.update({temp2: [Poarities[l][Aspects[l].index('DRINKS#PRICES')]]})	
        else:
            id11+=1
            temp = {}
            temp1 = str(id11)
            temp2 = int(temp1)
            dict11.update({temp2: [1]})
        if 'FOOD#STYLE_OPTIONS' in Aspects[l]:
            id12+=1
            temp = {}
            temp1 = str(id12)
            temp2 = int(temp1)
            dict12.update({temp2: [Poarities[l][Aspects[l].index('FOOD#STYLE_OPTIONS')]]})	
        else:
            id12+=1
            temp = {}
            temp1 = str(id12)
            temp2 = int(temp1)
            dict12.update({temp2: 1})
        temp1 = str(i)
        temp2 = int(temp1)    
        Dict1.update({temp2: dict1})
        Dict2.update({temp2: dict2})
        Dict3.update({temp2: dict3})
        Dict4.update({temp2: dict4})
        Dict5.update({temp2: dict5})
        Dict6.update({temp2: dict6})
        Dict7.update({temp2: dict7})
        Dict8.update({temp2: dict8})
        Dict9.update({temp2: dict9})
        Dict10.update({temp2: dict10})
        Dict11.update({temp2: dict11})
        Dict12.update({temp2: dict12})

 
###################### import dawid_skene ##################
        
responses = dawid_skene.generate_sample_data()
result1 = dawid_skene.run(Dict1)
result2 = dawid_skene.run(Dict2)
result3 = dawid_skene.run(Dict3)
result4 = dawid_skene.run(Dict4)
result5 = dawid_skene.run(Dict5)
result6 = dawid_skene.run(Dict6)
result7 = dawid_skene.run(Dict7)
result8 = dawid_skene.run(Dict8)
result9 = dawid_skene.run(Dict9)
result10 = dawid_skene.run(Dict10)
result11 = dawid_skene.run(Dict11)
result12 = dawid_skene.run(Dict12)

###################### BWM ##################

import pdb; pdb.set_trace() 
compared2best  = dict({'RESTAURANT#GENERAL':1, 'FOOD#QUALITY':2, 'RESTAURANT#MISCELLANEOUS':7,  'FOOD#PRICES':4, 'DRINKS#QUALITY':8, 'LOCATION#GENERAL':2, 'RESTAURANT#PRICES':5, 'AMBIENCE#GENERAL':1, 'DRINKS#STYLE_OPTIONS':3, 'SERVICE#GENERAL':3, 'DRINKS#PRICES':5 , 'FOOD#STYLE_OPTIONS':2});
compared2worst  = dict({'RESTAURANT#GENERAL':9, 'FOOD#QUALITY':5, 'RESTAURANT#MISCELLANEOUS':8,  'FOOD#PRICES':1, 'DRINKS#QUALITY':4, 'LOCATION#GENERAL':5, 'RESTAURANT#PRICES':5, 'AMBIENCE#GENERAL':9, 'DRINKS#STYLE_OPTIONS':8, 'SERVICE#GENERAL':7, 'DRINKS#PRICES':5 , 'FOOD#STYLE_OPTIONS':8});

w, zeta = calc_weight(compared2best, compared2worst)
print("Weights: ", w);


hotel_Scores = np.zeros(len(result1))
for i in range(len(result1)):
    hotel_Scores[i] = (w['AMBIENCE#GENERAL']*result1[i])+(w['DRINKS#PRICES']*result2[i])+(w['DRINKS#QUALITY']*result3[i])+(w['DRINKS#STYLE_OPTIONS']*result4[i])+(w['FOOD#PRICES']*result5[i])+(w['FOOD#QUALITY']*result6[i])+(w['FOOD#STYLE_OPTIONS']*result7[i])+(w['LOCATION#GENERAL']*result8[i])+(w['RESTAURANT#GENERAL']*result9[i])+(w['RESTAURANT#MISCELLANEOUS']*result10[i])+(w['RESTAURANT#PRICES']*result11[i])+(w['SERVICE#GENERAL']*result12[i])
                
############# Final #######################

print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
for i in range(len(hotel_Scores)):
    print("Hotel Scores ",i, ":, ", hotel_Scores[i]);


                
                