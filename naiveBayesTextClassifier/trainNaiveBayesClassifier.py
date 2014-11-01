# -*-coding:utf-8 -*-

import sys
import os
import numpy as np
from preprocess import getDocWordsList 
import cPickle as p #store python object into file

#stop words file path
global stopWordFileName
stopWordFileName = r"stoplist.txt"


#get vocabulary set from all the documents in "mini-newsgroup" directory
#for each document, pre-process(header-strip, stop word removal) is done.
#words whose count is less than 3 are deleted.
def getVocabulary(rootDir):
       #get all the file names (absolute path with names)
    fileList = [] 
    for parent, dirNames, fileNames in os.walk(rootDir):        
        for fileName in fileNames:
            fileList.append(os.path.join(parent, fileName))

    #spare file by file to get the vovabulary
    tempVocab = []
    for eachfile in fileList:
        #get a file's word list, pre-process is done.
        wordsList = getDocWordsList(eachfile, stopWordFileName)
        #add this file's words into vocabulary, combination(extend)
        tempVocab += wordsList
    tempVocab.sort()
    
##    #delete the words whose count is less than 3
##    #very time-consuming!!!
##    #tempVocab = [word for word in tempVocab if tempVocab.count(word) >= 3]
##    tempVocab = filter(lambda word: tempVocab.count(word) >= 3, tempVocab)
##    
##    #convert the vocabulary list to vocabulary set, remove reduplicative words
##    #and remain only one.
##    #the words in vocabulary is unique and sorted alphabetically(alphabetic order).
##    VocabSet = set(tempVocab) #first convert to set
##
##    Vocabulary = []
##    Vocabulary = list(VocabSet) #convert back to list
##    Vocabulary.sort()

    #delete the words whose count is less than 3, Another Method!
    vocabCountDict = {} #store word count for each word in tempVocab
    for word in tempVocab:
        if not vocabCountDict.has_key(word):
            vocabCountDict[word] = 1 # not in dict, add it into it
        else:
            vocabCountDict[word] += 1 #already in dict, count++

    for word,count in vocabCountDict.items():
        if (count < 3):
            del vocabCountDict[word] #delete word whose count < 3
    Vocabulary = vocabCountDict.keys() #final vocabulary
    
    #store the vocabulary into a file, to avoid repetitive function call
    vocabFile = "vocabulary.data"
    f = file(vocabFile, 'w')
    p.dump(Vocabulary, f) #dump the object into a file
    f.close()

    #vocabulary size is 12942 (remove words whose count<3), in mini_newsgroups
    #without removing low frequency words, vocab size will be about 30,000
    return Vocabulary  



#get all the class names of dataset in folder "mini_newsgroup"
def getClsNames(rootDir):
    clsNames = []
    clsNames = os.listdir(rootDir) #get class names in mini_newsgroup folder
    clsNames.sort() #sort alphabetically
    return clsNames



#calc prior class prior probility, using document ratio belonging to each class
#in this case, it can be set to {1/20, 1/20, ...., 1/20} because 20 classes have
#the same number of documents.
#the default rootDir is "mini_newsgroup"
def classPriorProbility(rootDir):
    clsList = []
    clsList = getClsNames(rootDir) #get class names

    #class prior probility vector (1.0/20, 1.0/20, ...., 1.0/20)
    #[priorProb]=[(x+1.0/20)]  is a list for list combination
    clsPriorProb = [[(x+1.0/20)] for x in np.zeros(len(clsList), np.float)]

    #constructing the (class, [probility]) dictionary ---- clsPriorProbDict
    clsPriorProbDict = {} #empty dict
    for i in range( len(clsList) ):
        clsPriorProbDict[clsList[i]] = clsPriorProb[i]

    #store the clsPriorProbDict into file using cPickle module
    clsPriorProbFile = "clsPriorProb.data"
    f = file(clsPriorProbFile, 'w')
    p.dump(clsPriorProbDict, f) #dump the object into a file
    f.close()
    
    #{class:[priorProb]} [priorProb]  is a list for list combination
    return clsPriorProbDict 



#combine the documents of one class into a bigger text document
#ATTENTION: leave out 20% of the documents for testing !!!
#input: root directory of data set, i.e. mini_newsgroups
#output: dict{"class name","class words list"}  20 classes
#the default rootDir is "mini_newsgroup"
def combineClsDocs(rootDir):
    #get class names in mini_newsgroup folder
    clsList = getClsNames(rootDir) 
        
    #get all the directory names with absolute path
    dirList = []
    for parent, dirNames, fileNames in os.walk(rootDir):        
        for dirName in dirNames:
            dirList.append(os.path.join(parent, dirName))
    dirList.sort() #sort alphabetically

    clsWordDict = {} #(class name : class words list) 20 pairs
    
    for i in range(len(clsList)): #20 classes
        #get all the file names in a class, store them into fileList
        fileList = [] # file names with absolute path of a class
        for parent, dirNames, fileNames in os.walk(dirList[i]):        
            for fileName in fileNames:
                fileList.append(os.path.join(parent, fileName))

        #combine the files of a class into clsWordsList, each file pre-processed
        trainSize = int(len(fileList)*0.8) #80% of the documents for training
        clsWordsList = []
        for j in range(trainSize):
            clsWordsList += getDocWordsList(fileList[j], stopWordFileName)

        #construct className:classWordsList dictionary
        #clsList[i] refers to the i-th class name
        #clsWordsList refers to the class words list respectively
        clsWordDict[ clsList[i] ] = clsWordsList

    return clsWordDict



#*******************************************************************************
#                       ML estimate p(Tj|Ck)
# Tj refers to the term in the Vocabulary, and Ck refers to the class type.
#*******************************************************************************
# m-estimate method is used, which has the following form:
# p(Tj|Ck) ≈ (Njk + 1)/(N+ sizeof(Vocabulary))
# N refers to the word counts of a class, which is represented as clsWordDict
# Njk refers to the occurences of term Tj in class k, which is represented as clsWordDict
#*******************************************************************************
#input: vocab       -- the vocabulary list stored in vocabulary.data file
#       clsWordDict -- the {class name : clsWordList} dict
#                      obtained from combineClsDocs(rootDir) function
#output: wordProbOnClsDict -- {class name : p(Tj|Ck) list} dictonary
def wordProbOnCls(vocab, clsWordDict):
    vocabSize = len(vocab) #vocabulary size is 12942
    
    #store (className : p(Tj|Ck) list ) pair in the wordProbOnClsDict dictionary
    wordProbOnClsDict = {}  
    
    for className, clsWordsList in clsWordDict.items():
        #integrated document(cls) size in word counts, includes repetitive words
        docWordCount = len(clsWordsList)

        #store p(Tj|Ck) list of a certain class
        #and the terms' locations are settled by vocabulary permanently
        wordProbOnClsList = []
        
        #calc p(Tj|Ck) list for a class and store them in wordProbOnClsList
        for j in range(vocabSize):
            #term Tj's count in a class, which is represented by clsWordsList
            occurences = clsWordsList.count(vocab[j])    
            #MLE of p(Tj|Ck) and append it to wordProbOnClsList
            wordProbOnCls = (occurences + 1.0)/(docWordCount + vocabSize)
            wordProbOnClsList.append(wordProbOnCls)

        #combine class name with wordProbOnClsList, form a wordProbOnClsDict
        #for all the 20 classes
        wordProbOnClsDict[className] = wordProbOnClsList
    
    #store the clsPriorProbDict into file using cPickle module
    wordProbOnClsFile = "wordProbOnCls.data"
    f = file(wordProbOnClsFile, 'w')
    p.dump(wordProbOnClsDict, f) #dump the object into a file
    f.close()
    
    return wordProbOnClsDict



if __name__ == "__main__":
    #get vocabulary from dataset
    vocabularyList = getVocabulary(sys.argv[1])
    print "size of vocabulary: ", len(vocabularyList)
    
    #get clsPriorProbDict, and store it in file "clsPriorProb.data"
    clsPriorProbDict = classPriorProbility(sys.argv[1])
    for clsName,clsPriorProb in clsPriorProbDict.items():
        print clsName, clsPriorProb

    #read vocabulary back from the storage file
    vocabFile = "vocabulary.data"
    f1 = file(vocabFile, 'r')
    vocabulary = []
    vocabulary = p.load(f1)
    f1.close()    
    print len(vocabulary) #vocabulary size is 12942, in mini_newsgroups

    #get clsWord dict
    clsWordDict = combineClsDocs(sys.argv[1])
    for clsName,wordsList in clsWordDict.items():
        print clsName, len(wordsList)
    print len(clsWordDict)
    
    #get wordProbOnClsDict, and store it in file "wordProbOnCls.data"
    wordProbOnClsDict = wordProbOnCls(vocabulary, clsWordDict)
    for clsName,wordProbOnClsList in wordProbOnClsDict.items():
        print clsName, len(wordProbOnClsList)
    print len(wordProbOnClsDict)    
##    wordProbOnClsList = wordProbOnClsDict.values()
##    print wordProbOnClsList[0]
