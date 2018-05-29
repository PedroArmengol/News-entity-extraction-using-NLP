


############ METHOD 2



#Pedro Armengol
#05/09/2018
#Initial exercise to locate source and target in the text
import os
import re
import spacy
import numpy
import re
import math
from random import shuffle
from spacy.attrs import ENT_IOB, ENT_TYPE

nlp = spacy.load('es')

###### Preprocessing
name = "AFP_SPA_19940707.0301"
# Functions
def adjust(num, text):
    '''
    Adjust the number of spaces based on the number of lines in the document
    '''
    spaces = text[0:num].count('\n')
    adj_number = num - spaces

    return adj_number


def format_text(name):
    '''
    Return dictionary with key: text_name and value(text,label entities)
    '''
    print(name)
    text = open('protest_annotation_chicago/raw/'+name+".txt")
    print("text")
    print(text)
    text = text.read()
    annotations = open('protest_annotation_chicago/ann/'+name+".ann")
    print("annotations")
    print(annotations)
    #Format the annotations
    list_tuples = []
    list_entities = []
    for line in annotations:
        stripped = line.strip() 
        stripped = stripped.split("\t")
        #Detect if entity (not rlationship R)
        print(stripped)
        if stripped[0][0] == "T":
            if ";" in stripped[1]:
                stripped[1] = stripped[1].replace(";"," ")
            #Extract position
            position = [int(s) for s in stripped[1].split() if s.isdigit()]
            #Extract class 
            entity = stripped[1].split()[0]
            list_entities.append(entity)
            #Make tupples depending on the number of times that entity appears
            for i in range(int(len(position)/2)):
                tuple1 = (adjust(position[0],text),adjust(position[1],text),entity)
                list_tuples.append(tuple1)
                #Removed append locations
                position.remove(position[0])
                position.remove(position[0])
  
    return ((text,{"entities": list_tuples}),(list_entities))


def train_data(list_documents,percentage):
    #Suffle elements of the list
    #Divide data in training and testing
    #apply format text to each document in training
    data = list_documents
    shuffle(data)
    #Rounded downward:more elements for the testing set
    training_set = data[0:math.floor(len(list_documents)*percentage)]
    testing_set = data[math.floor(len(list_documents)*percentage):]
    training_data = []
    list_entities = []
    for i in training_set:
        tuple1, entities = format_text(i)
        training_data.append(tuple1)
        for i in entities:
            list_entities.append(i)
    #Get a list with the type of entities in the text
    list_entities = set(list_entities)

    #Take out cases where there is not label data
    for i in training_data:
        if i[1]["entities"] == []:
            training_data.remove(i)

    return training_data, list_entities, testing_set

'''
RUN IT
'''

list_documents = os.listdir("protest_annotation_chicago/raw")
list1 = []
for i in list_documents:
    name = re.sub('\.txt$', '', i)
    list1.append(name)

list_documents = list1
training_data, list_entities, testing_set = train_data(list_documents,0.85)



