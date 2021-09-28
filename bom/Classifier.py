import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import ComplementNB
from matplotlib import pyplot as plt

# Alma 11:29
# WOM 1:16

file1 = open("./Lesson1.txt", "r")
train = file1.readlines()

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train[::2])
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = ComplementNB().fit(X_train_tfidf, np.array([1, 1, 1, 1, 1, 1, 1, 1,
                                                  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                                  3, 3, 3, 3, 3, 3, 3,
                                                  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                                  5, 5, 5, 5, 5, 5, 5]))

dfObj = pd.DataFrame(columns=['Book', 'Restoration', 'PoS', 'Gospel', 'Commandments', 'Laws'])
i = 0

verse = "NE3 18:20 And whatsoever ye shall ask the Father in my name, which is right, believing that ye shall receive, behold it shall be given unto you."
print(clf.predict(count_vect.transform([verse])))

# for filename in os.listdir('bom'):
#     file1 = open("./bom/" + filename, "r")
#     book = file1.readlines()
#     book = book[::2]
#     resto = 0
#     pos = 0
#     gospel = 0
#     comm = 0
#     laws = 0
#
#     for verse in book:
#         cat = clf.predict(count_vect.transform([verse]))
#         if 1 in cat:
#             resto = resto + 1
#         if 2 in cat:
#             pos = pos + 1
#         if 3 in cat:
#             gospel = gospel + 1
#         if 4 in cat:
#             comm = comm + 1
#         if 5 in cat:
#             laws = laws + 1
#
#     restoF = (resto / len(book))
#     posF = (pos / len(book))
#     gospelF = (gospel / len(book))
#     commF = (comm / len(book))
#     lawsF = (laws / len(book))
#
#     dfObj.loc[i] = [filename, restoF, posF, gospelF, commF, lawsF]
#     i = i + 1
#
# dfObj.plot(kind='bar', x='Book', y='Restoration',color='blue')
# dfObj.plot(kind='bar', x='Book', y='PoS',color='red')
# dfObj.plot(kind='bar', x='Book', y='Gospel',color='green')
# dfObj.plot(kind='bar', x='Book', y='Commandments',color='purple')
# dfObj.plot(kind='bar', x='Book', y='Laws',color='yellow')
# plt.show()