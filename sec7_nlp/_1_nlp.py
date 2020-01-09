import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')

reviews = pd.read_csv('restaurant_reviews.csv')
ps = PorterStemmer()

# remove non-chars
reviews['Review'] = reviews['Review'].map(lambda sentence: re.sub('[^A-Za-z]', ' ', sentence))
# to lower case
reviews['Review'] = reviews['Review'].map(lambda x: x.lower())
# split sentence to words
reviews['Review'] = reviews['Review'].map(lambda x: x.split())
# remove stop words in sentence
reviews['Review'] = reviews['Review'].map \
    (lambda words: [ps.stem(word) for word in words if word not in set(stopwords.words('english'))])
# re-generate sentence
reviews['Review'] = reviews['Review'].map(lambda words: ' '.join(words))

from sklearn.feature_extraction.text import CountVectorizer

# Feature extraction
# Bag of words (BOW)
cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(reviews['Review'].values).toarray()
y = reviews['Liked'].values


# Machine learning starts
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)

cm = confusion_matrix(y_pred, y_test)
print(cm)  # %72.5 accuracy

