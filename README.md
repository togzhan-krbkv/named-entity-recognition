# Project: NLP
## Предобработка данных


```python
!pip install nltk
import nltk
nltk.download()

import pandas as pd
pd.set_option('display.max_colwidth', 100)

data = pd.read_csv("SMSSpamCollection", sep='\t', header=None)
data.columns = ['label', 'body_text']

data.head()
```
**Вывод:**
| label | body_text |
|-------|-----------|
| ham | Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there g... |
| ham | Ok lar... Joking wif u oni... |
| spam | Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive ... |
| ham | U dun say so early hor... U c already then say... |
| ham | Nah I don't think he goes to usf, he lives around here though... |


## Punctuation (Пунктуация)
```python
import string
string.punctuation

def remove_punctuation(text):
    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    return text_nopunct

data['body_text_clean'] = data['body_text'].apply(lambda x: remove_punctuation(x))
data.head()
```
**Вывод:**
| label | body_text | body_text_clean |
|-------|-----------|-----------------|
| ham | Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there g... | Go until jurong point crazy Available only in bugis n great world la e buffet Cine there got amo... |
| ham | Ok lar... Joking wif u oni... | Ok lar Joking wif u oni |
| spam | Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive ... | Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005 Text FA to 87121 to receive e... |
| ham | U dun say so early hor... U c already then say... | U dun say so early hor U c already then say |
| ham | Nah I don't think he goes to usf, he lives around here though | Nah I dont think he goes to usf he lives around here though |


## Tokenisation (Токенизация)
```python
import re

def tokenise(text):
    tokens = re.split('\W+', text)
    return tokens

data['body_text_tokenised'] = data['body_text_clean'].apply(lambda x: tokenise(x.lower()))
data.head()
```
**Вывод:**
| label | body_text                                                                                                                                                              | body_text_clean                                                                                                                                                      | body_text_tokenised                                                                                                       |
|-------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| ham   | Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there g...                                                                    | Go until jurong point crazy Available only in bugis n great world la e buffet Cine there got amo...                                                                  | [go, until, jurong, point, crazy, available, only, in, bugis, n, great, world, la, e, buffet, ci...                      |
| ham   | Ok lar... Joking wif u oni...                                                                                                                                          | Ok lar Joking wif u oni                                                                                                                                              | [ok, lar, joking, wif, u, oni]                                                                                           |
| spam  | Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive ...                                                                    | Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005 Text FA to 87121 to receive e...                                                                  | [free, entry, in, 2, a, wkly, comp, to, win, fa, cup, final, tkts, 21st, may, 2005, text, fa, to...                      |
| ham   | U dun say so early hor... U c already then say...                                                                                                                      | U dun say so early hor U c already then say                                                                                                                          | [u, dun, say, so, early, hor, u, c, already, then, say]                                                                  |
| ham   | Nah I don't think he goes to usf, he lives around here though                                                                                                          | Nah I dont think he goes to usf he lives around here though                                                                                                          | [nah, i, dont, think, he, goes, to, usf, he, lives, around, here, though]                                                |


## Stemming (Стемминг)
```python
ps = nltk.PorterStemmer()

def stemming(tokenised_text):
    text = [ps.stem(word) for word in tokenised_text]
    return text

data['body_text_stemmed']  = data['body_text_nostop'].apply(lambda x: stemming(x))
data.head()
```
**Вывод:**
| label | body_text | body_text_clean | body_text_tokenised | body_text_nostop |
|-------|-----------|-----------------|---------------------|------------------|
| ham | Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there g... | Go until jurong point crazy Available only in bugis n great world la e buffet Cine there got amo... | [go, until, jurong, point, crazy, available, only, in, bugis, n, great, world, la, e, buffet, ci... | [go, jurong, point, crazy, available, bugis, n, great, world, la, e, buffet, cine, got, amore, wat] |
| ham | Ok lar... Joking wif u oni... | Ok lar Joking wif u oni | [ok, lar, joking, wif, u, oni] | [ok, lar, joking, wif, u, oni] |
| spam | Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive ... | Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005 Text FA to 87121 to receive e... | [free, entry, in, 2, a, wkly, comp, to, win, fa, cup, final, tkts, 21st, may, 2005, text, fa, to... | [free, entry, 2, wkly, comp, win, fa, cup, final, tkts, 21st, may, 2005, text, fa, 87121, receiv... |
| ham | U dun say so early hor... U c already then say... | U dun say so early hor U c already then say | [u, dun, say, so, early, hor, u, c, already, then, say] | [u, dun, say, early, hor, u, c, already, say] |
| ham | Nah I don't think he goes to usf, he lives around here though | Nah I dont think he goes to usf he lives around here though | [nah, i, dont, think, he, goes, to, usf, he, lives, around, here, though] | [nah, dont, think, goes, usf, lives, around, though] |

## Stopwords (Стоп-слова)
```python
import nltk

stopword = nltk.corpus.stopwords.words('english')

def remove_stopwords(tokenised_list):
    text = [word for word in tokenised_list if word not in stopword]
    return text

data['body_text_nostop'] = data['body_text_tokenised'].apply(lambda x: remove_stopwords(x))
data.head()
```
**Вывод:**
| label | body_text | body_text_clean | body_text_tokenised | body_text_nostop | body_text_stemmed |
|-------|-----------|-----------------|---------------------|------------------|-------------------|
| ham | Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there g... | Go until jurong point crazy Available only in bugis n great world la e buffet Cine there got amo... | [go, until, jurong, point, crazy, available, only, in, bugis, n, great, world, la, e, buffet, ci... | [go, jurong, point, crazy, available, bugis, n, great, world, la, e, buffet, cine, got, amore, wat] | [go, jurong, point, crazi, avail, bugi, n, great, world, la, e, buffet, cine, got, amor, wat] |
| ham | Ok lar... Joking wif u oni... | Ok lar Joking wif u oni | [ok, lar, joking, wif, u, oni] | [ok, lar, joking, wif, u, oni] | [ok, lar, joke, wif, u, oni] |
| spam | Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive ... | Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005 Text FA to 87121 to receive e... | [free, entry, in, 2, a, wkly, comp, to, win, fa, cup, final, tkts, 21st, may, 2005, text, fa, to... | [free, entry, 2, wkly, comp, win, fa, cup, final, tkts, 21st, may, 2005, text, fa, 87121, receiv... | [free, entri, 2, wkli, comp, win, fa, cup, final, tkt, 21st, may, 2005, text, fa, 87121, receiv,... |
| ham | U dun say so early hor... U c already then say... | U dun say so early hor U c already then say | [u, dun, say, so, early, hor, u, c, already, then, say] | [u, dun, say, early, hor, u, c, already, say] | [u, dun, say, earli, hor, u, c, alreadi, say] |
| ham | Nah I don't think he goes to usf, he lives around here though | Nah I dont think he goes to usf he lives around here though | [nah, i, dont, think, he, goes, to, usf, he, lives, around, here, though] | [nah, dont, think, goes, usf, lives, around, though] | [nah, dont, think, goe, usf, live, around, though] |

## Lemmatising (Лемматизация)
```python
wn = nltk.WordNetLemmatizer()

def lemmatizing(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text

data['body_text_lemmatized'] = data['body_text_nostop'].apply(lambda x: lemmatizing(x))
data.head()
```
**Вывод:**
| label | body_text | body_text_clean | body_text_tokenised | body_text_nostop | body_text_stemmed | body_text_lemmatized |
|-------|-----------|-----------------|---------------------|------------------|-------------------|----------------------|
| ham | Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there g... | Go until jurong point crazy Available only in bugis n great world la e buffet Cine there got amo... | [go, until, jurong, point, crazy, available, only, in, bugis, n, great, world, la, e, buffet, ci... | [go, jurong, point, crazy, available, bugis, n, great, world, la, e, buffet, cine, got, amore, wat] | [go, jurong, point, crazi, avail, bugi, n, great, world, la, e, buffet, cine, got, amor, wat] | [go, jurong, point, crazy, available, bugis, n, great, world, la, e, buffet, cine, got, amore, wat] |
| ham | Ok lar... Joking wif u oni... | Ok lar Joking wif u oni | [ok, lar, joking, wif, u, oni] | [ok, lar, joking, wif, u, oni] | [ok, lar, joke, wif, u, oni] | [ok, lar, joking, wif, u, oni] |
| spam | Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive ... | Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005 Text FA to 87121 to receive e... | [free, entry, in, 2, a, wkly, comp, to, win, fa, cup, final, tkts, 21st, may, 2005, text, fa, to... | [free, entry, 2, wkly, comp, win, fa, cup, final, tkts, 21st, may, 2005, text, fa, 87121, receiv... | [free, entri, 2, wkli, comp, win, fa, cup, final, tkt, 21st, may, 2005, text, fa, 87121, receiv,... | [free, entry, 2, wkly, comp, win, fa, cup, final, tkts, 21st, may, 2005, text, fa, 87121, receiv... |
| ham | U dun say so early hor... U c already then say... | U dun say so early hor U c already then say | [u, dun, say, so, early, hor, u, c, already, then, say] | [u, dun, say, early, hor, u, c, already, say] | [u, dun, say, earli, hor, u, c, alreadi, say] | [u, dun, say, early, hor, u, c, already, say] |
| ham | Nah I don't think he goes to usf, he lives around here though | Nah I dont think he goes to usf he lives around here though | [nah, i, dont, think, he, goes, to, usf, he, lives, around, here, though] | [nah, dont, think, goes, usf, lives, around, though] | [nah, dont, think, goe, usf, live, around, though] | [nah, dont, think, go, usf, life, around, though] |

## Vectorisation (Векторизация)