from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from pageparser import docset

docs = docset(sys.argv[1])

#Get the tfidf descriptor. 
#vectorizer = TfidfVectorizer(max_df=0.6, min_df=0.3)
vectorizer = TfidfVectorizer()

def cosine_similarity(doc1, doc2):
    tfidf = vectorizer.fit_transform([doc1, doc2])   #Automatically normalized. no additional effort
    return ((tfidf * tfidf.T).A)[0,1]                #Cosine similarity

#Add items to deduped list based on consine similarity
deduped = []
for idx, doc in docs:
    next_item = False
    if idx == 0:
        deduped.append(doc)
    else:
        tlist = []
        for item in deduped:
            if cosine_similarity(doc.getText(), item.getText()) > float(sys.argv[2]):
                next_item = True;
                break
        if next_item is True:
            continue
        else:
            deduped.append(doc)

print "Total items in training set:" + str(docs.getCount())
print "Remaining items after dedup: " + str(len(deduped))
