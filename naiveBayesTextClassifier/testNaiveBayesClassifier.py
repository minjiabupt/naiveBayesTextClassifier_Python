# -*-coding:utf-8 -*-

import sys  #for sys.argv[i]
import os   #for os.path
import math #for log(x)
import numpy as np #for ndarray and its numerical calculation
import cPickle as p
from preprocess import getDocWordsList

#stop words file path
global stopWordFileName
stopWordFileName = r"stoplist.txt"

#get text(document) features into a textFeatureList, each item represent a
#word count, word belongs to Vocabulary and words are in vocabulary order.
#input: text name with abusolute path
#output: a textFeatureList, each item represent a word count
def getTextFeature(docName, vocabulary):
    #get text words firt
    textWordsList = []
    textWordsList = getDocWordsList(docName, stopWordFileName)
    textWordsList.sort()

    #calc text feature in terms of Vocabulary
    textFeatureList = []
    for eachWord in vocabulary:
        textFeatureList.append(textWordsList.count(eachWord))

    #textFeatureVector = np.array(textFeatureList) #convet list to array
    #return textFeatureVector #len = sizeof(Vocabulary)
    return textFeatureList



#classify a text, using C = arg(k) max { f(Ck|Xj)= W(k)*Xj }
#where Ck is class type, and Xj is text j
#W(k)=( W0(k), W1(k), ..., Wm(k) )=( log(P(Ck)), log(P(Tj|Ck)) )  m=|Vocab|
#W(k) is log of ( P(Ck), wordProbOnClsList )
#Xj=( 1,textFeatureList ) where 1 corresponds to W0(k)=log(P(Ck))
#*************************************************************************
#input: textFeatureList -- word count list representing a text
#       clsPriorProbDict  -- {class name: P(Ck)} dict
#       wordProbOnClsDict -- {class name: P(Tj|Ck)} dict
#output: class tag of a text, string
def classifyText(textFeatureList, clsPriorProbDict, wordProbOnClsDict):
    textFeatureList = [1,] + textFeatureList #Xj=(1, textFeatureList)
    textFeatureVector = np.array(textFeatureList) #convet list to array

    #combine clsPriorProbDict with wordProbOnClsDict to form the compelete
    #weight vector W(k): {class name : [clsPriorProb wordProbOnClsList]}
    #log operation is calc later
    weightProbDict = {} #W(k) for 20 classes, {clsName:W(k)}
    for eachClass in clsPriorProbDict.keys():
        #combine 2 lists 
        weightProbDict[eachClass] = clsPriorProbDict[eachClass]+wordProbOnClsDict[eachClass]
        
    #class confidence(belonging prob) dict for decision
    clsConfidenceDict = {} #{class name:class confidence}

    #calc confidence for each class
    for eachClass, weightProbList in weightProbDict.items():
        #calc log of weightProbDict
        weightProbList = [np.float(math.log(item)) for item in weightProbList]
        #convet list to array
        weightProbVector = np.array(weightProbList)
        
        #calc W(k)*Xj using np.dot(a,b) operation
        clsConfidenceDict[eachClass] = np.dot(weightProbVector,textFeatureVector)
        
    #compare the class confidence and decision class Tag for document
    #key = lambda k: Dict[k]
    #refers to ordering fun of max(), i.e. order by Dict[k]'s value
##    print "class name \t\t class confidence"
##    for cls, clsConfidence in clsConfidenceDict.items():
##        print "%s \t\t %s" %(cls,clsConfidence)
##    print 
    clsTag=max(clsConfidenceDict.iterkeys(),key=lambda k:clsConfidenceDict[k])
    return clsTag #class name/tag of a document

        
        
#test the classifier using the whole test set, which is the documents of
#the last 20 docs of mini_newsgroup folder
#input:
#@1 rootDir -- "mini_newsgroup" folder containing the all the files,but only
#              last 20% of the files will be used to test the NB classifier
#@2 clsPriorProbDict  - clsPriorProb, can be read from file: "clsPriorProb.data"
#@3 wordProbOnClsDict - wordProbOnCls, can be read from file: "wordProbOnCls.data"
#@4 vocabulary       -- vocabulary list of the whole dataset
#output: precision of NB classifier
def testNB(rootDir, clsPriorProbDict, wordProbOnClsDict, vocabulary):
    #get all the directory names with absolute path
##    dirList = []
##    clsNames = [] #clsNames correspond with dirList, in the same order
    clsDirDict = {} #{class name : class file directory/path} dict
    for parent, dirNames, fileNames in os.walk(rootDir):        
        for dirName in dirNames:
            clsDirDict[dirName] = os.path.join(parent, dirName)
##            dirList.append(os.path.join(parent, dirName))
##            clsNames.append(dirName)
    
    #for the last 20% docs of each class(in dirList), using NB to get their classes,
    #and compare them with real class tags
    global totalErrorCount  #error classified document's total count
    global totalDocCount    #total document count
    totalDocCount = 0
    totalErrorCount = 0
    
    textFeatureList = [] #a document/text/file feature
    
    #{class name : class file directory/path} dict
    for clsName, clsDir in clsDirDict.items():
        #real class tag, documents in each subfolder are in the same class!
        realClassTag = clsName
        
        #get file names in each sub-folder
        fileList = [] 
        for parent, dirNames, fileNames in os.walk(clsDir):        
            for fileName in fileNames:
                fileList.append(os.path.join(parent, fileName))
        
        #classify each file of the last 20% documents in a class
        clsDocCount = len(fileList)
        testSize = int(0.2*clsDocCount)
        totalDocCount += testSize #total document count increase class by class        

        #last 20% files of the class [len(fileList)-testSize : len(fileList)]
        for j in range( (clsDocCount-testSize) , clsDocCount):
            #get a file's feature list
            textFeatureList = getTextFeature(fileList[j], vocabulary)
            textClsTag = classifyText(textFeatureList, clsPriorProbDict, wordProbOnClsDict)
            #decide weather a doc is classified correctly
            if textClsTag != realClassTag: #dirNames[i] refers to its real class
                totalErrorCount += 1
    print "total document count: %d" %totalDocCount
    print "error classified document count: %d" %totalErrorCount
    return 1.0*totalErrorCount/totalDocCount #error classification rate




if __name__ == "__main__":
    #read vocabulary back from the storage file
    vocabFile = "vocabulary.data"
    f1 = file(vocabFile, 'r')
    vocabulary = []
    vocabulary = p.load(f1)
    f1.close()    
    print len(vocabulary) #vocabulary size is 12942, in mini_newsgroups

    #read clsPriorProb back from the storage file
    clsPriorProbFile = "clsPriorProb.data"
    f2 = file(clsPriorProbFile, 'r')
    clsPriorProbDict = {}
    clsPriorProbDict = p.load(f2)
    f2.close()

    #read wordProbOnCls back from the storage file
    wordProbOnClsFile = "wordProbOnCls.data"
    f3 = file(wordProbOnClsFile, 'r')
    wordProbOnClsDict = {}
    wordProbOnClsDict = p.load(f3)
    f3.close()

    fileName = "51121"
    textFeatureList = getTextFeature(fileName, vocabulary)
    clsTag = classifyText(textFeatureList, clsPriorProbDict, wordProbOnClsDict)
    print "class tag of document %s is: %s" %(fileName, clsTag)
    
    errorRate = testNB(sys.argv[1], clsPriorProbDict, wordProbOnClsDict, vocabulary)
    print "Naive Bayes Classifier's precision is : %f" %(1 - errorRate)
    print "Naive Bayes Classifier's error rate is : %f" %errorRate
