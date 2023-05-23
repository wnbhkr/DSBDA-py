# #Assigment 7 Text processing and calculation of TF-IDF

# import nltk
# import re

# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("averaged_perceptron_tagger")
# nltk.download("omw-1.4")

# text ="Tokenization is the first step in text analytics. The process of breaking down a text par"

# #Sentence Tokenization.
# from nltk.tokenize import sent_tokenize
# tokenized_text=sent_tokenize(text)
# print(tokenized_text)

# #Word Tokenization
# from nltk.tokenize import word_tokenize
# tokenized_word=word_tokenize(text)
# print(tokenized_word)

# #Print stop words of English
# from nltk.corpus import stopwords
# stop_words=set(stopwords.words("english"))
# print(stop_words)

# text= "How to remove stop words with NLTK Library in Python?"
# text= re.sub("[^a-zA-Z]"," ",text)
# tokens = word_tokenize(text.lower ())
# filtered_text=[]

# for w in tokens:
#     if w not in stop_words:
#         filtered_text.append(w)
# print ("Tokenized Sentence:", tokens)
# print ("Filterd Sentence:", filtered_text)

# from nltk.stem import PorterStemmer
# e_words= ["wait", "waiting", "waited", "waits"]
# ps=PorterStemmer()
# for w in e_words:
#     rootWord=ps.stem(w)
# print(rootWord)

# from nltk.stem import WordNetLemmatizer
# wordnet_lemmatizer= WordNetLemmatizer()
# text = "studies studying cries cry"
# tokenization=nltk.word_tokenize(text)
# for w in tokenization:
#     print("Lemma for () is ".format(w, wordnet_lemmatizer, lemmatize[w]))

# from nltk.tokenize import word_tokenize
# data="The pink sweater fit her perfectly"
# words=word_tokenize(data)
# for word in words:
#     print(nltk.pos_tag([word]))

# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# d0="Jupiter is the largest Planet"
# d1="Mars is the fourth planet from the Sun"
# string=[d0,d1]
# tfidf =Tfidfvectorizer()
# result = tfidf.fit_transform(string)
# print("Word indices:", tfidf.vocabulary)
# print("TF-IDF Values:", result)


import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("omw-1.4")

# Sentence Tokenization
text = "Tokenization is the first step in text analytics. The process of breaking down a text par"
tokenized_text = sent_tokenize(text)
print(tokenized_text)

# Word Tokenization
tokenized_word = word_tokenize(text)
print(tokenized_word)

# Print stop words of English
stop_words = set(stopwords.words("english"))
print(stop_words)

# Remove stop words from text
text = "How to remove stop words with NLTK Library in Python?"
text = re.sub("[^a-zA-Z]", " ", text)
tokens = word_tokenize(text.lower())
filtered_text = [w for w in tokens if w not in stop_words]
print("Tokenized Sentence:", tokens)
print("Filtered Sentence:", filtered_text)

# Stemming
e_words = ["wait", "waiting", "waited", "waits"]
ps = PorterStemmer()
stemmed_words = [ps.stem(w) for w in e_words]
print(stemmed_words)

# Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
text = "studies studying cries cry"
tokenization = nltk.word_tokenize(text)
lemmatized_words = [wordnet_lemmatizer.lemmatize(w) for w in tokenization]
print("Lemmatized Words:", lemmatized_words)

# Part-of-Speech Tagging
data = "The pink sweater fit her perfectly"
words = word_tokenize(data)
pos_tags = nltk.pos_tag(words)
print(pos_tags)

# TF-IDF Calculation
d0 = "Jupiter is the largest Planet"
d1 = "Mars is the fourth planet from the Sun"
string = [d0, d1]
data = pd.Series(string)
tfidf = TfidfVectorizer()
result = tfidf.fit_transform(data)
print("Word indices:", tfidf.vocabulary_)
print("TF-IDF Values:", result)
