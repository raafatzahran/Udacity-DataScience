import pandas as pd
# Set a random seed
import random
random.seed(42)

# Implement the Bag of Words process from scratch!
documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

# Step 1: Convert all strings to their lower case form.
lower_case_documents = []
for i in documents:
    lower_case_documents.append(i.lower())
print(lower_case_documents)
# ---------------------------------------
# Step 2: Removing all punctuations
sans_punctuation_documents = []
import string

for i in lower_case_documents:
    sans_punctuation_documents.append(''.join(c for c in i if c not in string.punctuation))

print(sans_punctuation_documents)
# ---------------------------------------
# Step 3: Tokenization
preprocessed_documents = []
for i in sans_punctuation_documents:
    preprocessed_documents.append(i.split(' '))
print(preprocessed_documents)
# ---------------------------------------
# Step 4: Count frequencies
frequency_list = []
import pprint
from collections import Counter

for i in preprocessed_documents:
    frequency_list.append(Counter(i))

pprint.pprint(frequency_list)
# =========================================
'''
Here we will look to create a frequency matrix on a smaller document set to make sure we understand how the 
document-term matrix generation happens. We have created a sample document set 'documents'.
'''
documents = ['Hello, how are you!',
                'Win money, win from home.',
                'Call me now.',
                'Hello, Call hello you tomorrow?']
# Import the sklearn.feature_extraction.text.CountVectorizer method
# and create an instance of it called 'count_vector'.
from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()
print(count_vector)
# Fit your document dataset to the CountVectorizer object you have created using fit()
count_vector.fit(documents)
tokens = count_vector.get_feature_names()
print(tokens)
'''
Create a matrix with the rows being each of the 4 documents, 
and the columns being each word. The corresponding (row, column) 
value is the frequency of occurrence of that 
word(in the column) in a particular document(in the row).
'''
doc_array = count_vector.transform(documents).toarray()
print(doc_array)
frequency_matrix = pd.DataFrame(doc_array, columns=tokens)
print(frequency_matrix)
# =========================================
# Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
df = pd.read_table("../data/SMSSpamCollection", sep="\t", header=None, names=['label', 'sms_message'])
print(df.head(n=5))
df['label'] = df.label.map({'ham':0, 'spam':1})
print(df.shape)
print(df.describe())
print(df.groupby('label').describe())

# split into training and testing sets
# USE from sklearn.model_selection import train_test_split to avoid seeing deprecation warning.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                    df['label'],
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))

# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

predictions = naive_bayes.predict(testing_data)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))
