


############ METHOD 2



#Pedro Armengol
#05/09/2018
#Initial exercise to locate source and target in the text
import os
import spacy
import numpy
import re
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
    text = open('protest_annotation_chicago/raw/'+name+".txt")
    text = text.read()
    annotations = open('protest_annotation_chicago/ann/'+name+".ann")
    #Format the annotations
    list_tuples = []
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
    		#Make tupples depending on the number of times that entity appears
    		for i in range(int(len(position)/2)):
    			tuple1 = (adjust(position[0],text),adjust(position[1],text),entity)
    			list_tuples.append(tuple1)
    			#Removed append locations
    			position.remove(position[0])
    			position.remove(position[0])
  
    return (text,{"entities": list_tuples})


tuple1 = format_text(name)

'''
###################################

Train data
    data = []
    for i in list_docs:
    	obj = format_text(i)
    	data.append(obj)



list_documents = os.listdir("protest_annotation_chicago/raw")
list_annotations = os.listdir("protest_annotation_chicago/ann")
list_headers = []
list_places =[]

for i in list_documents:
    header, place = read_text(i)
    list_headers.append(header)
    list_places.append(place)
'''

'''
#POS
doc = nlp(list_headers[0])

#POS 
for token in doc:
	print(token.text, token.lemma_, token.pos_, 
		token.tag_, token.dep_,token.shape_, 
		token.is_alpha, token.is_stop)
'''

'''
#ENTITY RECOGNITION 
# Check actual entities: 
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
print(ents)

# Add new entities
doc = nlp(u"FB is hiring a new Vice President of global policy")
ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
print('Before', ents)
# the model didn't recognise "FB" as an entity :(

ORG = doc.vocab.strings[u'ORG']  # get hash value of entity label
fb_ent = Span(doc, 0, 1, label=ORG) # create a Span for the new entity
doc.ents = list(doc.ents) + [fb_ent]

ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
print('After', ents)

#Setting the characteristics of the entities
doc = nlp.make_doc(u'London is a big city in the United Kingdom.')
print('Before', list(doc.ents))  # []

header = [ENT_IOB, ENT_TYPE]
attr_array = numpy.zeros((len(doc), len(header)))
attr_array[0, 0] = 3  # 
#ENT_IOB = 3 Begining, 1 Inside, 2 Outside
#
#
attr_array[0, 1] = doc.vocab.strings[u'GPE']
doc.from_array(header, attr_array)
print('After', list(doc.ents))  # [London]
'''





