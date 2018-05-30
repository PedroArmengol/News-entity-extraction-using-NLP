


############ NLP



#Pedro Armengol
#05/30/2018
#Initial exercise to locate source and target in the text (among other classes)
from __future__ import unicode_literals, print_function
import os
import re
import spacy
import numpy
import re
import math
from spacy.attrs import ENT_IOB, ENT_TYPE
import plac
import random
from pathlib import Path

###### PREPROCESSING
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
    text = open('protest_annotation_chicago/raw/'+name+".txt")
    text = text.read()
    annotations = open('protest_annotation_chicago/ann/'+name+".ann")
    #Format the annotations
    list_tuples = []
    list_entities = []
    for line in annotations:
        stripped = line.strip() 
        stripped = stripped.split("\t")
        #Detect if entity (not rlationship R)
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


def splitting(list_documents, percentage):
    '''
    Divide the docs in training and testing based on the selected percentage
    '''
    data = list_documents
    random.shuffle(data)
    #Rounded downward:more elements for the testing set
    training_set = data[0:math.floor(len(list_documents)*percentage)]
    testing_set = data[math.floor(len(list_documents)*percentage):]

    return training_set, testing_set

def preprocessing(list_documents):
    '''
    read the documents (in format_text)
    apply desire format text to each document 
    Built data structure to run spacy model (HMM models)
    '''
    data = []
    list_entities = []
    for i in list_documents:
        tuple1, entities = format_text(i)
        data.append(tuple1)
        for i in entities:
            list_entities.append(i)
    #Get a list with the type of entities in the text
    list_entities = set(list_entities)
    #Take out cases where there is not label data
    for i in data:
        if i[1]["entities"] == []:
            data.remove(i)

    return data, list_entities


#THREE FUNCTIONS
## TESTING THE MODEL
## ACCURACY FUNCTION


###### FITTING THE MODEL
def training(data, model=None, new_model_name = None, output_dir=None, n_iter=20):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('es')  # create blank Language class
        print("Created blank 'es' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe('ner')
    for i in list_entities:    
        ner.add_label(i)   # add new entity label to entity recognizer
    if model is None:
        optimizer = nlp.begin_training()
    else:
        # Note that 'begin_training' initializes the models, so it'll zero out
        # existing entity types.
        optimizer = nlp.entity.create_optimizer()

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    print("training model")
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            losses = {}
            for text, annotations in data:
                nlp.update([text], [annotations], sgd=optimizer, drop=0.35,
                           losses=losses)
            print(losses)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)


def testing(data,output_dir=None):

    # test the saved model
    print("Loading from", output_dir)
    nlp = spacy.load(output_dir)
    list_performance = []
    list_performance_loc = []
    for tups in data:
        if tups[1]["entities"] != []:
            #Predict for each document
            doc = nlp(tups[0])
            list_tuples = []
            for ent in doc.ents:
                tuple1 = (ent.start_char, ent.end_char,ent.label_)
                list_tuples.append(tuple1)
            #Check accuracy overall
            num = 0
            num_loc = 0
            count_loc = 0
            for j in tups[1]["entities"]:
                if j[2] == "location":
                    count_loc += 1
                for i in list_tuples:
                     if j == i:
                        num += 1
                        if j[2] == "location":
                            num_loc += 1        
            #How many of the original entities wherecorrctly predicted
            num = num/(len(tups[1]["entities"]))
            num_loc = num_loc/count_loc
            list_performance.append(num)
            list_performance_loc.append(num_loc)
    #Average accuracy in each document
    accuracy = sum(list_performance) / float(len(list_performance))
    accuracy_loc = sum(list_performance_loc) / float(len(list_performance_loc))
    
    return accuracy, accuracy_loc

#displacy.serve(doc, style='ent')

if __name__ == '__main__':

    list_documents = os.listdir("protest_annotation_chicago/raw")
    list1 = []
    for i in list_documents:
        name = re.sub('\.txt$', '', i)
        list1.append(name)
    # Process data
    list_documents = list1[0:50]
    training_set, testing_set = splitting(list_documents, 0.85)
    training_data, list_entities = preprocessing(training_set)
    testing_data, list_entities = preprocessing(testing_set)
    #Train models
    training(data=training_data,new_model_name="test_1",output_dir="saved_models")
    #Test models
    accuracy, accuracy_loc = testing(data=testing_data,output_dir="saved_models")
    print("The accuracy of the model is: {0} %".format(round(accuracy*100,3)))
    print("The accuracy of the model for class location is: {0} %".format(round(accuracy_loc*100,3)))

