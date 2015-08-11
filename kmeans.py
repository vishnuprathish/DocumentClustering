import nltk
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from pageparser import docset
import sys

n_clusters = sys.argv[2]
docs = docset(sys.argv[1])
labels = docs.getLabels()

vectorizer = TfidfVectorizer(max_df=0.6, min_df=0.3)

tfidf = vectorizer.fit_transform(docs.getData())

#tune max_iter and n_init
km = KMeans(n_clusters = 3, init='k-means++', max_iter=100, n_init=100)
clusters = km.fit_predict(tfidf)

print "Clustering of docs:"
print [str(cluster) + ":" + label[:-1] for (cluster, label) in zip(clusters, labels)]
