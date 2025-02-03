import csv
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import re
import pandas as pd
import numpy as np

# For sorting dictionaries
import operator


from nltk.stem import PorterStemmer


reviews = []
rev = []
rec = []

nltk.download('stopwords')

set(stopwords.words('english'))


# A helper function that removes all the non ASCII characters
# from the given string. Retuns a string with only ASCII characters.
def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)



# LOADING AND CLEANiING DATA



# Creating a data structure for each review:
# clean:    The preprocessed string of characters

with open('C:/Users/HP/Desktop/BE PROJECT/Datasets/redmi6.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:

        review= dict()
        review['Review Title'] = row[0]
        review['Customer name'] = row[1]
        review['Rating'] = (row[2])
        review['Date'] = (row[3])
        review['Category'] = row[4]
        review['Comments'] = row[5]
        review['Useful'] = row[6]

        # Ignore repated reviews
        if re.match(r'^RT.*', review['Comments']):
            continue

        review['clean'] = review['Comments']

        # Remove all non-ascii characters
        review['clean'] = strip_non_ascii(review['clean'])
        # Remove all single characters
        review['clean']= re.sub(r'\s+[a-zA-Z]\s+', ' ', review['clean'])

        # Remove single characters from the start
        review['clean'] = re.sub(r'\^[a-zA-Z]\s+', ' ', review['clean']) 
        
        # Remove numbers
        review['clean'] = re.sub(r'\d+', '', review['clean'])
        
        # Remove punctuations
        import string
        review['clean'] = review['clean'].translate(str.maketrans('', '', string.punctuation))
        
        # Remove whitespaces
        review['clean'] = review['clean'].strip("")

        # Normalize case
        review['clean'] = review['clean'].lower()

        # Remove the hashtag symbol
        review['clean'] = review['clean'].replace(r'#', '')
         # Stemming
        review['clean'] = review['clean'].split()
    
        ps = PorterStemmer()
    
        review['clean'] = [ps.stem(word) for word in review['clean'] if not word in set(stopwords.words('english'))]
        review['clean'] = ' '.join(review['clean'])
       
        
        reviews.append(review)
        rev.append(review['clean'])
        rec.append(review['Comments'])


lexicon = dict()

# Read in the lexicon. 
with open('lexicon_easy.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        lexicon[row[0]] = int(row[1])

# Using lexicon to score reviews
for review in reviews:
    score = 0
    for word in review['clean'].split():
        if word in lexicon:
            score = score + lexicon[word]

    review['score'] = score
    if (score > 0):
        review['sentiment'] = 'positive'
    elif (score < 0):
        review['sentiment'] = 'negative'
    else:
        review['sentiment'] = 'neutral'

# Print out summary stats
total = float(len(reviews))
num_pos = sum([1 for t in reviews if t['sentiment'] == 'positive'])
num_neg = sum([1 for t in reviews if t['sentiment'] == 'negative'])
num_neu = sum([1 for t in reviews if t['sentiment'] == 'neutral'])
print ("Positive: %5d (%.1f%%)" % (num_pos, 100.0 * (num_pos/total)))
print ("Negative: %5d (%.1f%%)" % (num_neg, 100.0 * (num_neg/total)))
print ("Neutral:  %5d (%.1f%%)" % (num_neu, 100.0 * (num_neu/total)))

reviews_sorted = sorted(reviews, key=lambda k: k['score'])

print ("\n\nTOP NEGATIVE REVIEWS")

negative_reviews = [d for d in reviews_sorted if d['sentiment'] == 'negative']
for review in negative_reviews[0:280]:
    print ("score=%.2f, clean=%s" % (review['score'], review['clean']))
    
print ("\n\nTOP POSITIVE TWEETS")    
positive_reviews = [d for d in reviews_sorted if d['sentiment'] == 'positive']
for review in positive_reviews[-280:]:
    print ("score=%.2f, clean=%s" % (review['score'], review['clean']))
    
print ("\n\nTOP NEUTRAL TWEETS")
neutral_reviews = [d for d in reviews_sorted if d['sentiment'] == 'neutral']
for review in neutral_reviews[0:280]:
    print ("score=%.2f, clean=%s" % (review['score'], review['clean']))


import pandas as pd
import csv
df = pd.read_csv('C:/Users/HP/Desktop/BE PROJECT/Datasets/redmi6.csv',' rb',engine='python',delimiter = ',')

actualScore = df['Rating']
def partition(x):
    if x == '5.0 out of 5 stars':
        return '5'
    elif x == '4.0 out of 5 stars':
        return '4'
    elif x == '3.0 out of 5 stars':
        return '3'
    elif x == '2.0 out of 5 stars':
        return '2'
    elif x == '1.0 out of 5 stars':
        return '1'

star = actualScore.map(partition)
df['Rating'] = star

x = df['Comments']
y = df['Category']
#VECTORIZATION
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
X = vectorizer.fit_transform(x).toarray()   


# SPLITTING THE DATASET INTO TRAINING SET AND TESTING SET
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Support Vector Machine
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
svm = SVC(random_state=101)
svm.fit(X_train,y_train)
predsvm = svm.predict(X_test)
print("Confusion Matrix for Support Vector Machines:")
print(confusion_matrix(y_test,predsvm))
print("Score:",round(accuracy_score(y_test,predsvm)*100,2))
print("Classification Report:",classification_report(y_test,predsvm))      


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
labels = ['Category','Comments']
df = pd.DataFrame.from_records(neutral_reviews, columns=labels)
df1=df['Category']
df1.value_counts(normalize=True)

import seaborn as sns
#Bar plot for neutral reviews
labels = ['Others','Camera', 'Battery', 'Display', 'Delivery']
Category1 = [80, 6, 4, 4, 3]
plt.bar(labels, Category1, tick_label=labels, width=0.8, color=['blue', 'red', 'pink','orange', 'black'])
plt.xlabel('Categories')
plt.ylabel('Percentage of Counts')
plt.title('Distribution of Neutral Comments Using Bar Plot')
plt.show()
#pie chart for neutral reviews
color = ['g', 'r', 'b', 'y', 'orange']
plt.pie(Category1, labels=labels, colors=color,startangle=90, shadow=True, autopct='%1.2f%%', explode=(0.1, 0, 0, 0, 0))
plt.title('Distribution of Neutral Comments Using Pie Chart')
plt.legend()
plt.show()

labels = ['Category','Comments']
df = pd.DataFrame.from_records(negative_reviews, columns=labels)
df2=df['Category']
df2.value_counts(normalize=True)
#Bar plot for negative reviews
labels = ['Others','Camera', 'Battery', 'Display']
Category1 = [44,31,20, 3]
plt.bar(labels, Category1, tick_label=labels, width=0.8, color=['blue', 'red', 'pink','orange'])
plt.xlabel('Categories')
plt.ylabel('Percentage of Counts')
plt.title('Distribution of Negative Comments Using Bar Plot')
plt.show()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
#Pie Chat for negative reviews
color = ['green', 'r', 'b', 'y']
plt.pie(Category1, labels=labels, colors=color,startangle=90,shadow=True, autopct='%1.2f%%', explode=(0.1, 0, 0, 0))
plt.title('Distribution of Negative Comments Using Pie Chart')
plt.legend()
plt.show()

labels = ['Category','Comments']
df = pd.DataFrame.from_records(positive_reviews, columns=labels)
df3=df['Category']
df3.value_counts(normalize=True)
#Bar plot for positive reviews
labels = ['Others','Display', 'Battery', 'Camera','Delivery']
Category1 = [61, 16, 10, 7,2]
plt.bar(labels, Category1, tick_label=labels, width=0.8, color=['blue', 'red', 'pink','orange','purple'])
plt.xlabel('Categories')
plt.ylabel('Percentage of Counts')
plt.title('Distribution of Positive Comments Using Bar Plot')
plt.show()
#Pie chat for positive reviews
color = ['green', 'r', 'b', 'y','pink']
plt.pie(Category1, labels=labels, colors=color,startangle=90,shadow=True, autopct='%1.2f%%', explode=(0.1, 0, 0, 0,0))
plt.title('Distribution of Positive Comments Using Pie Chart')
plt.legend()
plt.show()

