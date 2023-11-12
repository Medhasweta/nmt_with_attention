
import pandas as pd
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import string

df = pd.read_csv('Data/mar.txt',encoding='utf-8', sep='	',  names=['English', 'Marathi', 'Attribution'])

df.head()

"""#### drop Attribution"""

df.drop(['Attribution'], axis=1, inplace=True)

df.info()

df.isna().sum()

"""## clean text

#### remove mutiple spaces
"""

df.English = df.English.apply(lambda x: " ".join(x.split()))
df.Marathi = df.Marathi.apply(lambda x: " ".join(x.split()))

"""#### lowercare only english characters beause marathi dont have lower and uppercaser"""

df.English = df.English.apply(lambda x: x.lower())

"""### Contraction to expansion of english text

##### this contraction dictionary is combination from lot of places
"""

with open("Data/contraction_expansion.txt", 'rb') as fp:
    contractions= pickle.load(fp)

def expand_contras(text):
    '''
    takes input as word or list of words
    if it is string and contracted it will expand it
    example:
    it's --> it is
    won't --> would not
    '''
    if type(text) is str:
        for key in contractions:
            value = contractions[key]
            text = text.replace(key, value)
        return text
    else:
        return text

df.sample(10)

xyz = "i'm don't he'll you'll"
expand_contras(xyz)

df.English = df.English.apply(lambda x: expand_contras(x))

df.sample(5)

"""#### remove all punctuations"""

translator= str.maketrans('','', string.punctuation)

df.English= df.English.apply(lambda x: x.translate(translator))
df.Marathi= df.Marathi.apply(lambda x: x.translate(translator))

df.sample(5)

"""### Remove digits"""

import re

df.English= df.English.apply(lambda x: re.sub(r'[\d]+','', x))
df.Marathi= df.Marathi.apply(lambda x: re.sub(r'[\d]+','', x))

"""## Visualize some features of dataset

#### create new column for count of words
"""

df['en_word_count']= df.English.apply(lambda x: len(x.split()))
df['mar_word_count']= df.Marathi.apply(lambda x: len(x.split()))

"""#### create new column for count of characters"""

df['mar_char_count']= df.Marathi.apply(lambda x: len("".join(x.split())))
df['en_char_count']= df.English.apply(lambda x: len("".join(x.split())))

df.head()

plt.figure(figsize=(15,10))
sns.kdeplot(x=df.en_word_count, shade=True, color='blue', label='Real')

"""## note lot of sentences are of 4 to 7 length"""

max(df.en_word_count)

plt.figure(figsize=(15,10))
sns.kdeplot(x=df.mar_word_count, shade=True, color='green', label='Real')

max(df.mar_word_count)

plt.figure(figsize=(10,8))
sns.distplot(x=df.en_char_count)

plt.figure(figsize=(10,8))
sns.distplot(x=df.mar_char_count)

"""## Plot wordcloud"""

def plot_word_cloud(data):
    words=""
    for sent in data:
        sent= str(sent)
        sent=sent.lower()
        tokens= sent.split()
        words +=" ".join(tokens)+" "
    plt.figure(figsize=(15,12))
    wordcloud= WordCloud(width=800,height=800, background_color='aqua').generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')

plot_word_cloud(df.English)

"""### save cleaned text"""

df.head()

df.to_csv("Data/cleaned.csv",index=None)

"""# Conclusion

* 41028 samples of sentenses
* Min len of both eng and mar sentence is 1
* Max len of both is 35 -- this will help for padding
* And we cleaned text removed all punctuatuins digits and expanded contractions in this notebook
"""

