## module for pre-process the documents in text classification
##              module list
##    getStopWords:    read stop words list from a file 
##    stripHeader:     strip the header in the documents for classification
##    splitFile:       split the file into word list, deleting special symbols
##    removeStopWords: remove the stop words in the documents for classification


# -*- coding:utf-8 -*-
import sys
import re # regular expression


#************************* function ******************************************
#function for getting stopwords from a file, return a list
#************************** parameters ***************************************
#input: file name
#output: a list storing stopWordsList
#*****************************************************************************
def getStopWords(fileName):
    pf = open (fileName, 'r')
    stopWordsList = []
    for line in pf:
        line = line.strip()
        stopWordsList.append(line)
    pf.close()    
    return stopWordsList


#************************* function ******************************************
#strip the header in the documents,return the remaining contents in string
#
#header is defined as the firt big block in the file, which is seperated from
#the remaining content by a blank line.
#************************** parameters ***************************************
#input: file name
#output: a string of the file content without header
#*****************************************************************************
def stripHeader(fileName):
    pf = open(fileName, 'r')
    line = pf.readline()
    while (line != "\n"): #read until a blank line (blank line returns a '\n')
        line = pf.readline()

    #read the remaining content of the file
    strList = [] #string list for storing each line(a string)
    line = pf.readline().lower() #convert str to lower case
    while (line != ""): #read to the EOF, return an empty line
        strList.append(line)
        line = pf.readline().lower()
    pf.close()

    content = ""
    content = ''.join(strList) #construct string list, then join them
    #print content
    return content


#************************* function ******************************************
#split the file(with header stripped) into word list, return a word list
#************************** parameters ***************************************
#input: a string, containing the content of file
#output: a word list , with special symbols deleted
#*****************************************************************************
def splitFile(fileContent):
    # regular expression includes '\s' '\W' and blank.
    #\s match any blank character, equals to [\t\n\r\f\v]
    #\W refers to 'neither number nor letter', equals to [^a-zA-Z0-9_]
    expr = '\s+| |\W+' 
    wordList = []
    wordList = re.split(expr, fileContent)

    #delete digit items and mix items, remain only alpha words
    #using list comprehesion (** method 1 **)
    wordList = [item for item in wordList if item.isalpha()] 
    return wordList


#************************* function ******************************************
#strip the stop words in a word list, return a list
#************************** parameters ***************************************
#input: a list, containing the words of a file
#output: a word list , with stop words deleted
#*****************************************************************************
def removeStopWords(fileWordList, stopWordList):
    #using filter (** method 2 **)
    fileWordList = filter(lambda item: item not in stopWordList, fileWordList)
            
    #for i in range(0,1,len(fileWordList)): #out of range error
    #can be revised by reverse order (** method 3 **)
##    for i in range(len(fileWordList)-1, -1, -1):
##        if fileWordList[i] in stopWordList: 
##            del fileWordList[i]
    return fileWordList


#get document representation using words, pre-process is done.
#pre-process includes 1.header strip  2.remove stop words
def getDocWordsList(docName, stopWordFileName):
    tempWords = splitFile(stripHeader(docName))    
    stopWords = getStopWords(stopWordFileName)
    wordsList = removeStopWords(tempWords, stopWords)
    return wordsList



if __name__ == "__main__":
    fileContent = stripHeader(sys.argv[1])
    wordList = splitFile(fileContent)
    for each in wordList:
        print each,
    
    stopWords = getStopWords(sys.argv[2])
    wordListNoSW = removeStopWords(wordList, stopWords)
    print
    print
    for each in wordListNoSW:
        print each,

    
