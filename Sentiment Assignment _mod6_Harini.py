#!/usr/bin/env python
# coding: utf-8

# # ADS 509 Sentiment Assignment
# 
# This notebook holds the Sentiment Assignment for Module 6 in ADS 509, Applied Text Mining. Work through this notebook, writing code and answering questions where required. 
# 
# In a previous assignment you put together Twitter data and lyrics data on two artists. In this assignment we apply sentiment analysis to those data sets. If, for some reason, you did not complete that previous assignment, data to use for this assignment can be found in the assignment materials section of Blackboard. 
# 

# ## General Assignment Instructions
# 
# These instructions are included in every assignment, to remind you of the coding standards for the class. Feel free to delete this cell after reading it. 
# 
# One sign of mature code is conforming to a style guide. We recommend the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). If you use a different style guide, please include a cell with a link. 
# 
# Your code should be relatively easy-to-read, sensibly commented, and clean. Writing code is a messy process, so please be sure to edit your final submission. Remove any cells that are not needed or parts of cells that contain unnecessary code. Remove inessential `import` statements and make sure that all such statements are moved into the designated cell. 
# 
# Make use of non-code cells for written commentary. These cells should be grammatical and clearly written. In some of these cells you will have questions to answer. The questions will be marked by a "Q:" and will have a corresponding "A:" spot for you. *Make sure to answer every question marked with a `Q:` for full credit.* 
# 

# # Github:https://github.com/harinigautham/Ads_mod6

# In[1]:


get_ipython().run_line_magic('cd', 'C://Users//gauth//Documents//Harini//San Diego University//applied text mining//M1 Assignment Data//M1 Results')


# In[2]:


import os
import re
import emoji
import pandas as pd
import numpy as np

from collections import Counter, defaultdict
from string import punctuation

from nltk.corpus import stopwords

sw = stopwords.words("english")


# In[3]:


# Add any additional import statements you need here
import seaborn as sns

# Some punctuation variations
punctuation = set(punctuation) # speeds up comparison
tw_punct = punctuation - {"#"}

# Stopwords
#from textbook
sw = set(stopwords.words('english'))

# Two useful regex
whitespace_pattern = re.compile(r"\s+")
hashtag_pattern = re.compile(r"^#[0-9a-zA-Z]+")

# It's handy to have a full set of emojis
all_language_emojis = set()

for country in emoji.UNICODE_EMOJI : 
    for em in emoji.UNICODE_EMOJI[country] : 
        all_language_emojis.add(em)

        
# and now our functions
def descriptive_stats(tokens, num_tokens = 5, verbose=True) :
    """
        Given a list of tokens, print number of tokens, number of unique tokens, 
        number of characters, lexical diversity, and num_tokens most common
        tokens. Return a list of 
    """

    # Place your Module 2 solution here
    num_tokens = len(tokens)
    num_unique_tokens = len(set(tokens))
    num_characters = sum(len(token) for token in tokens)
    lexical_diversity = num_unique_tokens / num_tokens
    return(0)


    
def contains_emoji(s):
    
    s = str(s)
    emojis = [ch for ch in s if emoji.is_emoji(ch)]

    return(len(emojis) > 0)


def remove_stop(tokens) :
    # modify this function to remove stopwords
    #from textbook
    return [t for t in tokens if t.lower() not in sw]
 
def remove_punctuation(text, punct_set=tw_punct) : 
    return("".join([ch for ch in text if ch not in punct_set]))

def tokenize(text) : 
    """ Splitting on whitespace rather than the book's tokenize function. That 
        function will drop tokens like '#hashtag' or '2A', which we need for Twitter. """
    
    # modify this function to return tokens
    text = text.split()
    return(text)

def prepare(text, pipeline) : 
    tokens = str(text)
    
    for transform in pipeline : 
        tokens = transform(tokens)
        
    return(tokens)


# In[4]:


# change `data_location` to the location of the folder on your machine.
#data_location = "/users/chandler/dropbox/teaching/repos/ads-tm-api-scrape/"
data_location = "C://Users//gauth//Documents//Harini//San Diego University//applied text mining//M1 Assignment Data//M1 Results"
# These subfolders should still work if you correctly stored the 
# data from the Module 1 assignment
twitter_folder = "C://Users//gauth//Documents//Harini//San Diego University//applied text mining//M1 Assignment Data//M1 Results//twitter" 
lyrics_folder = "C://Users//gauth//Documents//Harini//San Diego University//applied text mining//M1 Assignment Data//M1 Results//lyrics"

positive_words_file = "positive-words.txt"
negative_words_file = "negative-words.txt"
tidy_text_file = "tidytext_sentiments.txt"


# ## Data Input
# 
# Now read in each of the corpora. For the lyrics data, it may be convenient to store the entire contents of the file to make it easier to inspect the titles individually, as you'll do in the last part of the assignment. In the solution, I stored the lyrics data in a dictionary with two dimensions of keys: artist and song. The value was the file contents. A Pandas data frame would work equally well. 
# 
# For the Twitter data, we only need the description field for this assignment. Feel free all the descriptions read it into a data structure. In the solution, I stored the descriptions as a dictionary of lists, with the key being the artist. 
# 
# 
# 

# In[5]:


# Read in the lyrics data
df_lyrics = {
    'artist': [],
    'song_name': [],
    'contents': []
}

for artist_folder in os.listdir(lyrics_folder):
    for song_file in os.listdir(os.path.join(lyrics_folder, artist_folder)):
        with open(os.path.join(lyrics_folder, artist_folder, song_file), 'r') as f:
            song_lyrics = f.read()
        df_lyrics['artist'].append(artist_folder)
        df_lyrics['song_name'].append(song_file.replace('www_azlyrics_comkcijojo_', '').replace('www_azlyrics_comsammhenshaw_', '').replace('.txt', ''))
        df_lyrics['contents'].append(song_lyrics)
        
df_lyrics = pd.DataFrame(df_lyrics)
df_lyrics.head()


# In[6]:


# Read in the twitter data
#Reading in data
import os
import pandas as pd

twitter_folder ="C://Users//gauth//Documents//Harini//San Diego University//applied text mining//M1 Assignment Data//M1 Results//twitter"   # Replace with the actual path

data = {
    'artist': [],
    'description': []
}

for filename in os.listdir(twitter_folder):
    if 'data' in filename and filename.endswith('.txt'):
        artist = filename.split('_')[0]
        with open(os.path.join(twitter_folder, filename), 'r', encoding='utf-8') as f:
            for line in f:
                fields = [t.strip() for t in line.split('\t') if t.strip()]
                if len(fields) >= 2:  # Check for the expected number of fields
                    description = fields[-1]
                    if description == 'description':
                        continue
                    data['artist'].append(artist)
                    data['description'].append(description)

df_twitter = pd.DataFrame(data)
df_twitter.head()


# In[7]:


my_pipeline = [str.lower, remove_punctuation, tokenize, remove_stop]
df_lyrics["tokens"] = df_lyrics['contents'].apply(prepare, pipeline=my_pipeline)
df_twitter["tokens"] = df_twitter["description"].apply(prepare, pipeline=my_pipeline)
df_twitter['has_emoji'] = df_twitter["description"].apply(contains_emoji)


# In[8]:


get_ipython().run_line_magic('pwd', '')


# In[9]:


# Read in the positive and negative words and the
# tidytext sentiment. Store these so that the positive
# words are associated with a score of +1 and negative words
# are associated with a score of -1. You can use a dataframe or a 
# dictionary for this.

#with open('negative-words.txt', 'r') as datafile:
    #negative_lines = datafile.readlines()
with open("C://Users//gauth//Documents//Harini//San Diego University//applied text mining//M1 Assignment Data//M1 Results//negative-words.txt", 'r') as datafile:
    negative_lines = datafile.readlines()
    
negative_words = []
for line in negative_lines:
    if ';' not in line and line !='\n':
        negative_words.append(line.strip())
        
with open('positive-words.txt', 'r') as datafile:
    positive_lines = datafile.readlines()

positive_words = []
for line in positive_lines:
    if ';' not in line and line !='\n':
        positive_words.append(line.strip())
        
df_tidytext = pd.read_csv('tidytext_sentiments.txt', sep='\t')

df_sentiment = {
    'word':[],
    'sentiment':[]
}

[df_sentiment['word'].append(word) for word in negative_words]
[df_sentiment['sentiment'].append('negative') for i in range(len(negative_words))]

[df_sentiment['word'].append(word) for word in positive_words]
[df_sentiment['sentiment'].append('positive') for i in range(len(positive_words))]

[df_sentiment['word'].append(word) for word in df_tidytext['word']]
[df_sentiment['sentiment'].append(sent) for sent in df_tidytext['sentiment']]

df_sentiment = pd.DataFrame(df_sentiment)
df_sentiment['score'] = np.where(df_sentiment['sentiment']=='negative', -1, 1)


# In[10]:


df_sentiment


# In[11]:


# your code here

def sentiment_score(tokens, df):
    score = 0
    for token in tokens:
        for idx in df.loc[(df['word'] == token)].index.tolist():
            score += df['score'][idx]
    return score


# ## Sentiment Analysis on Songs
# 
# In this section, score the sentiment for all the songs for both artists in your data set. Score the sentiment by manually calculating the sentiment using the combined lexicons provided in this repository. 
# 
# After you have calculated these sentiments, answer the questions at the end of this section.
# 

# In[12]:


df_lyrics['score'] = df_lyrics['tokens'].apply(lambda x: sentiment_score(x, df_sentiment))


# In[13]:


df_lyrics


# In[14]:


df_lyrics.groupby('artist')['score'].mean()
print("Artist has the higher average sentiment per song:", df_lyrics.groupby('artist')['score'].mean())


# In[15]:


lowest_song = df_lyrics[df_lyrics['artist']=='cher'].sort_values('score')[:3]
highest_song = df_lyrics[df_lyrics['artist']=='cher'].sort_values('score', ascending=False)

for idx, row in lowest_song.iterrows():
    print("Top 3 lowest sentiments song of Cher artist:", row['song_name'].replace("cher_", ""))
    print(row['contents'])
    print('-'*100)
    
print('\n')
for song in highest_song.iterrows():
    print("Top 3 highest sentiments song of Cher artist:", row['song_name'].replace("cher_", ""))
    print(row['contents'])
    print('-'*100)


# In[16]:


lowest_song = df_lyrics[df_lyrics['artist']=='robyn'].sort_values('score')[:3]
highest_song = df_lyrics[df_lyrics['artist']=='robyn'].sort_values('score', ascending=False)

for idx, row in lowest_song.iterrows():
    print("Top 3 lowest sentiments song of Robyn artist:", row['song_name'].replace("robyn_", ""))
    print(row['contents'])
    print('-'*100)
    
print('\n')
for song in highest_song.iterrows():
    print("Top 3 highest sentiments song of Robyn artist:", row['song_name'].replace("robyn_", ""))
    print(row['contents'])
    print('-'*100)


# ### Questions
# 
# Q: Overall, which artist has the higher average sentiment per song? 
# 
# A: Cher
# 
# ---
# 
# Q: For your first artist, what are the three songs that have the highest and lowest sentiments? Print the lyrics of those songs to the screen. What do you think is driving the sentiment score? 
# 
# A: The lyric of song content some keywords is driving the sentiment score
# 
# ---
# 
# Q: For your second artist, what are the three songs that have the highest and lowest sentiments? Print the lyrics of those songs to the screen. What do you think is driving the sentiment score? 
# 
# A: The lyric of song content some keywords is driving the sentiment score
# 
# ---
# 
# Q: Plot the distributions of the sentiment scores for both artists. You can use `seaborn` to plot densities or plot histograms in matplotlib.
# 
# 
# 

# In[17]:


import matplotlib.pyplot as plt
# Set up the figure and axis
fig, ax = plt.subplots(figsize = (12,6))

# Plot histograms using matplotlib
ax.hist(df_lyrics.loc[df_lyrics['artist'] == 'robyn', 'score'].values, alpha=0.5, label='robyn')
ax.hist(df_lyrics.loc[df_lyrics['artist'] == 'cher', 'score'].values, alpha=0.5, label='cher')

# Set the labels and title
ax.set_xlabel('Sentiment Score')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Sentiment Scores')

# Show the legend
ax.legend()

# Display the plot
plt.show()


# ## Sentiment Analysis on Twitter Descriptions
# 
# In this section, define two sets of emojis you designate as positive and negative. Make sure to have at least 10 emojis per set. You can learn about the most popular emojis on Twitter at [the emojitracker](https://emojitracker.com/). 
# 
# Associate your positive emojis with a score of +1, negative with -1. Score the average sentiment of your two artists based on the Twitter descriptions of their followers. The average sentiment can just be the total score divided by number of followers. You do not need to calculate sentiment on non-emoji content for this section.

# In[18]:


def tweet_score(s, df):
    s = str(s)
    emojis = [ch for ch in s if emoji.is_emoji(ch)]
    score = 0
    for token in emojis:
        for idx in df.loc[(df['word'] == token)].index.tolist():
            score += df['score'][idx]
    return score


# In[19]:


# your code here
positive_emoji = ['😂', '❤️', '😍', '😘', '💕', '😊', '💜', '💓', '🙃', '👏🏻', '💘', '🥰']
negative_emoji = ['😭', '😔', '😌', '🤫', '😩', '🤢', '😱', '😪', '😢', '💔', '😈', '😥']
emo = positive_emoji + negative_emoji
score = []
[score.append(1) for i in range(len(positive_emoji))]
[score.append(-1) for i in range(len(negative_emoji))]

df_emoji = pd.DataFrame({
    'word': emo,
    'score': score
})

df_twitter_tmp = df_twitter[df_twitter['has_emoji']==True]
df_twitter_tmp['score'] = df_twitter_tmp['tokens'].apply(lambda x: tweet_score(x, df_emoji))


# In[20]:


df_twitter_tmp


# In[21]:


print("The average sentiment of two artists:")
df_twitter_tmp.groupby('artist')['score'].mean()


# In[22]:


df_emoji_positive = pd.DataFrame({
    'word': positive_emoji,
    'score': [score.append(1) for i in range(len(positive_emoji))]
})

df_emoji_negative = pd.DataFrame({
    'word': negative_emoji,
    'score': [score.append(-1) for i in range(len(positive_emoji))]
})

df_emoji_positive['cher'] = 0
df_emoji_positive['robyn'] = 0
df_emoji_negative['cher'] = 0
df_emoji_negative['robyn'] = 0

df_twitter_emoji = df_twitter_tmp[df_twitter_tmp['score']!=0]
df_twitter_cher = df_twitter_emoji[df_twitter_emoji['artist']=='cher']
df_twitter_robyn = df_twitter_emoji[df_twitter_emoji['artist']=='robynkonichiwa']


# In[23]:


for idx, row in df_twitter_cher.iterrows():
    emojis = [ch for ch in row['tokens'] if emoji.is_emoji(ch)]
    if len(emojis)>0:
        for token in emojis:
            for idx in df_emoji_positive.loc[(df_emoji_positive['word'] == token)].index.tolist():
                df_emoji_positive['cher'][idx] += 1
                
for idx, row in df_twitter_cher.iterrows():
    emojis = [ch for ch in row['tokens'] if emoji.is_emoji(ch)]
    if len(emojis)>0:
        for token in emojis:
            for idx in df_emoji_negative.loc[(df_emoji_negative['word'] == token)].index.tolist():
                df_emoji_negative['cher'][idx] += 1


# In[24]:


for idx, row in df_twitter_robyn.iterrows():
    emojis = [ch for ch in row['tokens'] if emoji.is_emoji(ch)]
    if len(emojis)>0:
        for token in emojis:
            for idx in df_emoji_positive.loc[(df_emoji_positive['word'] == token)].index.tolist():
                df_emoji_positive['robyn'][idx] += 1
                
for idx, row in df_twitter_robyn.iterrows():
    emojis = [ch for ch in row['tokens'] if emoji.is_emoji(ch)]
    if len(emojis)>0:
        for token in emojis:
            for idx in df_emoji_negative.loc[(df_emoji_negative['word'] == token)].index.tolist():
                df_emoji_negative['robyn'][idx] += 1


# In[25]:


df_emoji_negative.sort_values('cher', ascending=False)[:1].drop('robyn', axis=1)


# In[26]:


df_emoji_positive.sort_values('cher', ascending=False)[:1].drop('robyn', axis=1)


# In[27]:


df_emoji_negative.sort_values('robyn', ascending=False)[:1].drop('cher', axis=1)


# In[28]:


df_emoji_positive.sort_values('robyn', ascending=False)[:1].drop('cher', axis=1)


# Q: What is the average sentiment of your two artists? 
# 
# A:  cher              0.125744
#     robynkonichiwa    0.068976
# 
# ---
# 
# Q: Which positive emoji is the most popular for each artist? Which negative emoji? 
# 
# A: most positive cher: 💜. robyn: 💜 .Most negative for both artist: 😈
# 
# 
