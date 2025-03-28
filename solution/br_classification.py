import pandas as pd
import numpy as np
import re
import math
from sklearn.linear_model import LogisticRegression
from scipy.stats import mannwhitneyu

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords



def remove_html(text):
    """Remove HTML tags using a regex."""
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    """Remove emojis using a regex pattern."""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  # enclosed characters
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

NLTK_stop_words_list = stopwords.words('english')
custom_stop_words_list = ['...']  # You can customize this list as needed
final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list

def remove_stopwords(text):
    """Remove stopwords from the text."""
    return " ".join([word for word in str(text).split() if word not in final_stop_words_list])

def clean_str(string):
    """
    Clean text by removing non-alphanumeric characters,
    and convert it to lowercase.
    """
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

import os
import subprocess
project = 'tensorflow'
path = f'./datasets/{project}.csv'

pd_all = pd.read_csv(path)
pd_all = pd_all.sample(frac=1, random_state=999)  # Shuffle

# Merge Title and Body into a single column; if Body is NaN, use Title only
pd_all['Title+Body'] = pd_all.apply(
    lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
    axis=1
)

# Keep only necessary columns: id, Number, sentiment, text (merged Title+Body)
pd_tplusb = pd_all.rename(columns={
    "Unnamed: 0": "id",
    "class": "sentiment",
    "Title+Body": "text"
})
pd_tplusb.to_csv('Title+Body.csv', index=False, columns=["id", "Number", "sentiment", "text"])



datafile = 'Title+Body.csv'

REPEAT = 10

out_csv_name = f'./{project}_NB.csv'

data = pd.read_csv(datafile).fillna('')
text_col = 'text'

original_data = data.copy()

# Text cleaning
data[text_col] = data[text_col].apply(remove_html)
data[text_col] = data[text_col].apply(remove_emoji)
data[text_col] = data[text_col].apply(remove_stopwords)
data[text_col] = data[text_col].apply(clean_str)

params = {
   'C': [0.01, 0.01, 0.1, 1, 10, 100]
}

# Lists to store metrics across repeated runs
accuracies  = []
precisions  = []
recalls     = []
f1_scores   = []
auc_values  = []

for repeated_time in range(REPEAT):
    # --- Split into train/test ---
    indices = np.arange(data.shape[0])
    train_index, test_index = train_test_split(
        indices, test_size=0.2, random_state=repeated_time
    )

    train_text = data[text_col].iloc[train_index]
    test_text = data[text_col].iloc[test_index]

    y_train = data['sentiment'].iloc[train_index]
    y_test  = data['sentiment'].iloc[test_index]

    #  TF-IDF vectorization 
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=1000
    )
    X_train = tfidf.fit_transform(train_text)
    X_test = tfidf.transform(test_text) 

    model = LogisticRegression(class_weight='balanced')
    # Logistic regression model & GridSearch
    grid = GridSearchCV(
        model,
        params,
        cv=5,              
        scoring='roc_auc'  
    )
    grid.fit(X_train, y_train)
    # Retrieve the best model
    best_clf = grid.best_estimator_
    best_clf.fit(X_train, y_train)
   
    y_pred = best_clf.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    # Precision (macro)
    prec = precision_score(y_test, y_pred, average='macro')
    precisions.append(prec)

    # Recall (macro)
    rec = recall_score(y_test, y_pred, average='macro')
    recalls.append(rec)

    # F1 Score (macro)
    f1 = f1_score(y_test, y_pred, average='macro')
    f1_scores.append(f1)

    # AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
    auc_val = auc(fpr, tpr)
    auc_values.append(auc_val)

final_accuracy  = np.mean(accuracies)
final_precision = np.mean(precisions)
final_recall    = np.mean(recalls)
final_f1        = np.mean(f1_scores)
final_auc       = np.mean(auc_values)

print("=== Logistic Regression + TF-IDF Results ===")
print(f"Number of repeats:     {REPEAT}")
print(f"Average Accuracy:      {final_accuracy:.4f}")
print(f"Average Precision:     {final_precision:.4f}")
print(f"Average Recall:        {final_recall:.4f}")
print(f"Average F1 score:      {final_f1:.4f}")
print(f"Average AUC:           {final_auc:.4f}")

try:
    # Attempt to check if the file already has a header
    existing_data = pd.read_csv(out_csv_name, nrows=1)
    header_needed = False
except:
    header_needed = True

df_log = pd.DataFrame(
    {
         # change these values to accuracy, recall, precision etc. to print the average values to csv rather then arrays in the csv.
        'repeated_times': [REPEAT],
        'Accuracy': [accuracies],
        'Precision': [precisions],
        'Recall': [recalls],
        'F1': [f1_scores],
        'AUC': [auc_values],
        'CV_list(AUC)': [str(auc_values)]
    }
)

df_log.to_csv(out_csv_name, mode='a', header=header_needed, index=False)

print(f"\nResults have been saved to: {out_csv_name}")


'''
this was how i did the statistical tests:
# 2) After the final metric calculations:
baseline_scores =  [0.59375, 0.5488095238095239, 0.5354037267080746, 0.4457070707070707, 0.6296296296296297, 0.5496894409937888, 0.5897435897435898, 0.5543478260869565, 0.5436081242532855, 0.5805288461538461]
solution_scores = precisions

stat, p = mannwhitneyu(baseline_scores, solution_scores, alternative='two-sided')

print("Mann-Whitney U test (baseline recall vs. solution recall):")
print(f"  U-statistic = {stat:.4f}, p-value = {p}")
'''

