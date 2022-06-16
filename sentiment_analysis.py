import pandas as pd

# Read in the dataset titled 'IMDB Dataset.csv' containing two columns 'review' and 'sentiment'
df_review = pd.read_csv('IMDB Dataset.csv')
df_review
#print(df_review)

# Because the dataset contains 50000 rows, we use a smaller sample to train the model faster
df_positive = df_review[df_review['sentiment']=='positive'][:15000]
df_negative = df_review[df_review['sentiment']=='negative'][:1666]

# There is significantly more positive entries being used so we have an imbalanced dataset
df_review_imb = pd.concat([df_positive, df_negative])

#print(df_review_imb)

# Due to the imbalance we have the choice to under sample the positive reviews or oversample the negative reviews
# In this case we will under sample the positive reviews
from imblearn.under_sampling import RandomUnderSampler
# Create a new instance of RandomUnderSampler (rus)
rus = RandomUnderSampler(random_state=0)
# Resample the unbalanced dataset by fitting rus where x represents that data that needs to be sampled and y corresponds to the label for each sample
df_review_bal, df_review_bal['sentiment']=rus.fit_resample(df_review_imb[['review']],
                                            df_review_imb['sentiment'])
df_review_bal
# After this x and y should be balanced and below we can compare the two datasets

#print(df_review_imb.value_counts('sentiment'))
#print(df_review_bal.value_counts('sentiment'))

'''
OUTPUT:
        positive    15000
        negative    1666

        negative    1666
        positive    1666

    As we can see, the dataset is now equally distributed
'''
# From here we need a train set and a test set to compare the model and provide an unbiased evaluation
# The data is split by allotting 33% to the test data set and the rest to the train dataset
from sklearn.model_selection import train_test_split

train, test = train_test_split(df_review_bal, test_size = 0.33,
                                random_state = 42)
# Set the independent and dependent variables within train and test
train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']

# Text Representation: For this sample we use the bag of words technique due to our prioritization of the frequency of words
# (TF-IDF) is used to 'weigh' how important a particular word that is used is to the classification
# The TF-IDF value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words = 'english')
train_x_vector = tfidf.fit_transform(train_x)
train_x_vector

pd.DataFrame.sparse.from_spmatrix(train_x_vector,
                                    index = train_x.index,
                                    columns = tfidf.get_feature_names())

test_x_vector = tfidf.transform(test_x)

# In order to fit an SVM model, we introduce the input as text reviews and output as sentiment
from sklearn.svm import SVC

svc = SVC(kernel = 'linear')
svc.fit(train_x_vector, train_y)

#In order to fit an decision tree model, we introduce the input as text reviews and output as sentiment
from sklearn.tree import DecisionTreeClassifier

dec_tree = DecisionTreeClassifier()
dec_tree.fit(train_x_vector, train_y)

# In order to fit Naive Bayes model, we introduce the input as text reviews and output as sentiment
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(train_x_vector.toarray(), train_y)

# In order to fit logistic regression model, we introduce the input as text reviews and output as sentiment
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(train_x_vector, train_y)

# We then use .score to obtain the mean accuracy of each model
#svc.score('Test samples', 'True Labels')
svc.score(test_x_vector, test_y)
dec_tree.score(test_x_vector, test_y)
gnb.score(test_x_vector.toarray(), test_y)
log_reg.score(test_x_vector, test_y)

'''
print(svc)
print(dec_tree)
print(gnb)
print(log_reg)

After printing each of them, we obtain the mean accuracy.

SVM: 0.84
Decision tree: 0.64
Naive Bayes: 0.63
Logistic Regression: 0.83

Here we see that SVM has the highest accuracy, so we will use that to see how other metrics perform
'''

# F1 is the weighted average of precision and recall (F1 is useful in imbalanced datasets)
# F1 Score = 2*(Recall * Precision) / (Recall + Precision)
from sklearn.metrics import f1_score
f1_score(test_y, svc.predict(test_x_vector),
        labels = ['positive', 'negative'],
        average = None)

from sklearn.metrics import classification_report

print(classification_report(test_y,
                            svc.predict(test_x_vector),
                            labels = ['positive', 'negative']))

'''
With a classification report we can see the classification metrics including ones calculated before
               precision    recall  f1-score   support

    positive       0.83      0.87      0.85       335
    negative       0.85      0.82      0.83       325

    accuracy                           0.84       660
   macro avg       0.84      0.84      0.84       660
weighted avg       0.84      0.84      0.84       660
'''

# The confusion matrix typically has two rows and two columns that report the number of false positives, false negatives, true positives, and true negatives
from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(test_y,
                            svc.predict(test_x_vector),
                            labels = ['positive', 'negative'])

from sklearn.model_selection import GridSearchCV
#set the parameters
parameters = {'C': [1,4,8,16,32] ,'kernel':['linear', 'rbf']}
svc = SVC()
svc_grid = GridSearchCV(svc,parameters, cv = 5)

svc_grid.fit(train_x_vector, train_y)

print(svc_grid.best_params_)
print(svc_grid.best_estimator_)