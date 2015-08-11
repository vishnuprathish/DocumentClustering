from bs4 import BeautifulSoup
import urllib2
import nltk
from stemming.porter2 import stem
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sys

class article:
    data = ""
    tokens = ""

    #Initialize, clean, parse, stem
    def __init__(self, url):
        self._tf_idf  = None
        print "Fetching:" + url
        req = urllib2.Request(url)
        response = urllib2.urlopen(req)
        rawHtml = response.read()
        self.data = self.html2Text(rawHtml)
        strips = """\\.!?,(){}[]"'"""
        self.tokens = [stem(c.strip(strips)) for c in self.data.lower().split()]
        #Stop words are removed during vectorization

    def html2Text(self, htmlData):
        soup = BeautifulSoup(htmlData, 'html.parser')
        return soup.get_text()

    def getText(self):
        data = " ".join(self.tokens)
        return data

    def getTokens(self):
        return self.tokens

    #tf idf calculation. Not used. 
    def tf_idf(self, cached=True):
        if self._tf_idf and cached:
            return self._tf_idf
        self._tf_idf = {}
        idf = self.idf()
        for w in self.tf.keys():
            self._tf_idf[w] = idf[w] * self.tf[w]
        return self._tf_idf

class docset:
    def __init__(self, file = None):
        self.index = 0
        self.docs = []
        self.current = 0 
        self.labels = []

        if file is not None:
            try:
                with open(file, "r") as fp:
                    for dataitem in fp:
                        try:
                            uri = dataitem.split(' ')
                        except ValueError:
                            print "Invalid file format"
                            exit(0)
                        self.add(article(uri[0]))
                        self.labels.append(uri[1])
            except IOError as e:
                print "Unable to open file"
                exit(0)

    def add(self, article):
        self.docs.append((self.index, article))
        self.index += 1

    def __iter__(self):
        return self

    def getLabels(self):
        return self.labels

    def next(self):
        if self.current >= self.index:
            raise StopIteration
        else:
            self.current += 1
            return self.docs[self.current-1]

    def getData(self):
        return [d[1].getText() for d in self.docs]

    def getCount(self):
        return self.index
