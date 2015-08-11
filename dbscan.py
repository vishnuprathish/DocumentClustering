from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from pageparser import article, docset
import sys

docs = docset(sys.argv[1])

for dataitem in open("data.txt", "r"):
    uri = dataitem.split(' ')
    docs.add(article(uri[0]))
    labels = [0, 1, 2]

vectorizer = TfidfVectorizer(max_df=0.6, min_df=0.3)

tfidf = vectorizer.fit_transform(docs.getData())

X = (tfidf * tfidf.T).A #cosine
db_a = DBSCAN(eps=0.3, min_samples=4).fit(X)

lab = db_a.labels_
print lab
print [str(cluster) + ":" + label[:-1] for (cluster, label) in zip(lab, docs.getLabels())]
