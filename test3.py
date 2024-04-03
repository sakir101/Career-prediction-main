# -*- coding: utf-8 -*-
"""

Original file is located at
    https://colab.research.google.com/drive/11DnNS4PSWZ-w3djMEw3JxKejydMEWu5P

## **Mounting Drive**
"""

from google.colab import drive
drive.mount('/content/drive')

"""# **Import All Libraries**"""

import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score



from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

"""## **DataSet**"""

# Specify the path to your Excel file
excel_file_path = '/content/drive/MyDrive/intellicruit/test3Dataset.xlsx'

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_file_path)

# Set options to display all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


df.head(4)

# Shuffle the DataFrame rows
df = df.sample(frac=1, random_state=42)

# Reset the index of the shuffled DataFrame
df = df.reset_index(drop=True)

df.head(4)

"""## **Import nltk, re**"""

nltk.download('stopwords')

"""## **import stopwords and enlish stopword show**"""

stop_words = stopwords.words('english')

"""## **User defined function to remove stopword and make word lower**"""

def remove_stopwords(content):
    con = re.sub('[^a-zA-Z]', ' ', content)
    con = con.lower()
    con = con.split()
    con = [word for word in con if not word in stopwords.words('english')]
    con = ' '.join(con)
    return con

df['Skill'] = df['Skill'].apply(remove_stopwords)

"""## **Labelizing career**"""

le_x= LabelEncoder()
df.Career = le_x.fit_transform(df.Career)

"""## **Show decode value**"""

unique_classes = le_x.classes_
for label, encoded_value in zip(unique_classes, range(len(unique_classes))):
    print(f"{label} is encoded as {encoded_value}")

# Mapping encoded labels to their respective strings
labels = {
    0: 'Artificial Intelligence',
    1: 'Data Science',
    2: 'Development',
    3: 'Security',
    4: 'Software Development and Engineering',
    5: 'User Experience (UX) and User Interface (UI) Design'
}

# Count occurrences of each label
label_counts = df['Career'].value_counts()

# Calculate percentage for each label
percentages = (label_counts / len(df)) * 100

# Generate labels for the pie chart (with count and percentage)
labels_for_plot = [f"{labels[label]} - {label_counts[label]} ({percentages[label]:.1f}%)"
                   for label in label_counts.index]

# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(label_counts, labels=labels_for_plot, autopct='', startangle=140)
plt.title('Percentage of Career Labels')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Mapping encoded labels to their respective strings
labels = {
    0: 'Artificial Intelligence',
    1: 'Data Science',
    2: 'Development',
    3: 'Security',
    4: 'Software Development and Engineering',
    5: 'User Experience (UX) and User Interface (UI) Design'
}

# Define the percentages for train-test split
train_percentage = 0.8
test_percentage = 0.2

# Empty DataFrames to store train and test splits
train_data = pd.DataFrame()
test_data = pd.DataFrame()

# Iterate through each class
for label, total_count in labels.items():
    # Filter rows for the current label
    subset = df[df['Career'] == label]

    # Perform train-test split for the subset
    train_subset, test_subset = train_test_split(subset, train_size=train_percentage, test_size=test_percentage)

    # Append train and test splits to respective DataFrames
    train_data = train_data.append(train_subset)
    test_data = test_data.append(test_subset)

# Shuffle the DataFrame rows
train_data = train_data.sample(frac=1, random_state=42)

# Reset the index of the shuffled DataFrame
train_data = train_data.reset_index(drop=True)

# Shuffle the DataFrame rows
test_data = test_data.sample(frac=1, random_state=42)

# Reset the index of the shuffled DataFrame
test_data = test_data.reset_index(drop=True)

"""# **Taking Input**"""

x_train = train_data['Skill']
x_test = test_data['Skill']

"""## **Taking output class in y**"""

y_train = train_data['Career']
y_test = test_data['Career']

"""## **Apply TfidfVectorizer**"""

vect=TfidfVectorizer()
x_train=vect.fit_transform(x_train)
x_test=vect.transform(x_test)

"""## **Apply Decission Tree Classifier**"""

model=DecisionTreeClassifier()
model.fit(x_train, y_train)
prediction=model.predict(x_test)
prediction
model.score(x_test, y_test)

# Initialize the model
model = DecisionTreeClassifier()

# Perform cross-validation
scores = cross_val_score(model, x_train, y_train, cv=5)  # 5-fold cross-validation
average_accuracy = scores.mean()
print("Average Cross-Validation Accuracy:", average_accuracy)

# Generate confusion matrix
cm = confusion_matrix(y_test, prediction)
print(cm)
# Display the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

report = classification_report(y_test, prediction)
print("Classification Report:\n", report)

weighted_metrics = precision_recall_fscore_support(y_test, prediction, average='weighted')

# Display the weighted precision, recall, and f1-score
weighted_precision = weighted_metrics[0]
weighted_recall = weighted_metrics[1]
weighted_f1 = weighted_metrics[2]

print(f"Weighted Precision: {weighted_precision}")
print(f"Weighted Recall: {weighted_recall}")
print(f"Weighted F1-Score: {weighted_f1}")

p_dec = round(weighted_precision * 100, 2)
r_dec = round(weighted_recall * 100, 2)
f1_dec = round(weighted_f1 * 100, 2)

"""## **Apply Support Vector**"""

# Set the random_state when creating the SVC classifier
svm_classifier = SVC(kernel='linear', probability=True)

# Fit the classifier on the training data
svm_classifier.fit(x_train, y_train)

# Make predictions on the test data
y_pred = svm_classifier.predict(x_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy:", accuracy)

# Initialize the SVM classifier with desired parameters (e.g., linear kernel)
svm_classifier = SVC(kernel='linear', probability=True)

# Perform cross-validation to estimate the accuracy
# Adjust the number of folds (cv parameter) based on your preference
# Here, 5-fold cross-validation is used
scores = cross_val_score(svm_classifier, x_train, y_train, cv=5)

# Calculate the average accuracy across all folds
average_accuracy = scores.mean()
print("Average Cross-Validation Accuracy:", average_accuracy)

cm = confusion_matrix(y_test, y_pred)
print(cm)
# Display the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

weighted_metrics = precision_recall_fscore_support(y_test, y_pred, average='weighted')

# Display the weighted precision, recall, and f1-score
weighted_precision = weighted_metrics[0]
weighted_recall = weighted_metrics[1]
weighted_f1 = weighted_metrics[2]

print(f"Weighted Precision: {weighted_precision}")
print(f"Weighted Recall: {weighted_recall}")
print(f"Weighted F1-Score: {weighted_f1}")

p_svc = round(weighted_precision * 100, 2)
r_svc = round(weighted_recall * 100, 2)
f1_svc = round(weighted_f1 * 100, 2)

unique_classes = le_x.classes_
for label, encoded_value in zip(unique_classes, range(len(unique_classes))):
    print(f"{label} is encoded as {encoded_value}")

# Save the trained model for later use

joblib.dump(svm_classifier, 'svm_model.pkl')
joblib.dump(vect, 'tfidf_vectorizer.pkl')

print("Model trained and saved.")

"""## **Model generate for support vector**"""

# Load the trained model and TF-IDF vectorizer
svm_classifier = joblib.load('svm_model.pkl')
vect = joblib.load('tfidf_vectorizer.pkl')


def preprocess_text(content):
    # Remove non-alphabet characters and convert to lowercase
    content = re.sub('[^a-zA-Z]', ' ', content).lower()
    # Tokenize the text
    tokens = content.split()
    # Remove stopwords
    tokens = [word for word in tokens if not word in stopwords.words('english')]
    # Join the tokens back into a string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def career(user_data):
    # Preprocess the user_data
    preprocessed_data = preprocess_text(user_data)
    # Vectorize the preprocessed_data (assuming you have 'vect' defined)
    input_data = [preprocessed_data]
    print("Input Data:", input_data)
    vector_form = vect.transform(input_data)

    # Make predictions using your SVM classifier (assuming you have 'svm_classifier' defined)
    # This assumes 'svm_classifier' supports probability estimation (e.g., SVC with probability=True)
    class_probabilities = svm_classifier.predict_proba(vector_form)

    # Get the classes (labels) for your SVM classifier
    classes = svm_classifier.classes_

    # Create a dictionary to store class probabilities
    class_prob_dict = {}
    for i, class_label in enumerate(classes):
        class_prob_dict[class_label] = class_probabilities[0][i]

    # Find the class with the highest probability
    predicted_class = max(class_prob_dict, key=class_prob_dict.get)

    return class_prob_dict, predicted_class

def career1(user_data):
  vector_form = vect.transform([user_data])
  class_probabilities = svm_classifier.predict_proba(vector_form)
  classes = svm_classifier.classes_

    # Create a dictionary to store class probabilities
  class_prob_dict = {}
  for i, class_label in enumerate(classes):
        class_prob_dict[class_label] = class_probabilities[0][i]

    # Find the class with the highest probability
  predicted_class = max(class_prob_dict, key=class_prob_dict.get)

  return class_prob_dict, predicted_class

"""## **Taking input**"""

result, predicted_class = career1("""Artificial Intelligence (AI) and Machine Learning, Data Analysis and Visualization, Statistical Analysis, Research on Artificial Intelligence and Machine learning""")

"""## **Predict Output**"""

print("Class Probabilities:")
for class_label, probability in result.items():
    print(f"{class_label}: {probability * 100:.2f}%")
print(f"Predicted Class: {predicted_class}")

"""## **Apply Logistic Regression**"""

# Create a Logistic Regression classifier
logistic_classifier = LogisticRegression(random_state=10)

# Fit the classifier on the training data
logistic_classifier.fit(x_train, y_train)

# Make predictions on the test data
y_pred = logistic_classifier.predict(x_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy:", accuracy)

# Create a Logistic Regression classifier
logistic_classifier = LogisticRegression(random_state=10)

# Perform cross-validation to estimate the accuracy
# Adjust the number of folds (cv parameter) based on your preference
# Here, 5-fold cross-validation is used
scores = cross_val_score(logistic_classifier, x_train, y_train, cv=5)

# Calculate the average accuracy across all folds
average_accuracy = scores.mean()
print("Average Cross-Validation Accuracy:", average_accuracy)

cm = confusion_matrix(y_test, y_pred)
print(cm)
# Display the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

weighted_metrics = precision_recall_fscore_support(y_test, y_pred, average='weighted')

# Display the weighted precision, recall, and f1-score
weighted_precision = weighted_metrics[0]
weighted_recall = weighted_metrics[1]
weighted_f1 = weighted_metrics[2]

print(f"Weighted Precision: {weighted_precision}")
print(f"Weighted Recall: {weighted_recall}")
print(f"Weighted F1-Score: {weighted_f1}")

p_lr = round(weighted_precision * 100, 2)
r_lr = round(weighted_recall * 100, 2)
f1_lr = round(weighted_f1 * 100, 2)

"""## **Apply KNN**"""

# Create a KNN classifier with a specified number of neighbors (e.g., n_neighbors=3)
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier on the training data
knn_classifier.fit(x_train, y_train)

# Make predictions on the test data
y_pred = knn_classifier.predict(x_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy:", accuracy)

# Create a KNN classifier with a specified number of neighbors (e.g., n_neighbors=3)
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Perform cross-validation to estimate the accuracy
# Adjust the number of folds (cv parameter) based on your preference
# Here, 5-fold cross-validation is used
scores = cross_val_score(knn_classifier, x_train, y_train, cv=5)

# Calculate the average accuracy across all folds
average_accuracy = scores.mean()
print("Average Cross-Validation Accuracy:", average_accuracy)

cm = confusion_matrix(y_test, y_pred)
print(cm)
# Display the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

weighted_metrics = precision_recall_fscore_support(y_test, y_pred, average='weighted')

# Display the weighted precision, recall, and f1-score
weighted_precision = weighted_metrics[0]
weighted_recall = weighted_metrics[1]
weighted_f1 = weighted_metrics[2]

print(f"Weighted Precision: {weighted_precision}")
print(f"Weighted Recall: {weighted_recall}")
print(f"Weighted F1-Score: {weighted_f1}")

p_knn = round(weighted_precision * 100, 2)
r_knn = round(weighted_recall * 100, 2)
f1_knn = round(weighted_f1 * 100, 2)

"""## **Apply Naive Bayes**"""

# Create a Multinomial Naive Bayes classifier
naive_bayes_classifier = MultinomialNB()

# Fit the classifier on the training data
naive_bayes_classifier.fit(x_train, y_train)

# Make predictions on the test data
y_pred = naive_bayes_classifier.predict(x_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy:", accuracy)

# Create a Multinomial Naive Bayes classifier
naive_bayes_classifier = MultinomialNB()

# Perform cross-validation to estimate the accuracy
# Adjust the number of folds (cv parameter) based on your preference
# Here, 5-fold cross-validation is used
scores = cross_val_score(naive_bayes_classifier, x_train, y_train, cv=5)

# Calculate the average accuracy across all folds
average_accuracy = scores.mean()
print("Average Cross-Validation Accuracy:", average_accuracy)

cm = confusion_matrix(y_test, y_pred)
print(cm)
# Display the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

weighted_metrics = precision_recall_fscore_support(y_test, y_pred, average='weighted')

# Display the weighted precision, recall, and f1-score
weighted_precision = weighted_metrics[0]
weighted_recall = weighted_metrics[1]
weighted_f1 = weighted_metrics[2]

print(f"Weighted Precision: {weighted_precision}")
print(f"Weighted Recall: {weighted_recall}")
print(f"Weighted F1-Score: {weighted_f1}")

p_nb = round(weighted_precision * 100, 2)
r_nb = round(weighted_recall * 100, 2)
f1_nb = round(weighted_f1 * 100, 2)

"""## **Import Necessary Libraries**"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Dropout, GlobalMaxPooling1D, Reshape

# Set a random seed for reproducibility
# Set seeds for reproducibility
seed_value = 42
tf.random.set_seed(seed_value)
np.random.seed(seed_value)

"""## **Split, Create Tensorflow dataset, defne batch size and shuffle the data**"""

# import numpy as np
# # Split the training data into training and validation sets
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=52)

# # Create TensorFlow datasets
# train_dataset = tf.data.Dataset.from_tensor_slices((x_train.toarray(), y_train))
# val_dataset = tf.data.Dataset.from_tensor_slices((x_val.toarray(), y_val))
# test_dataset = tf.data.Dataset.from_tensor_slices((x_test.toarray(), y_test))

# # Define batch size
# batch_size = 32

# # Shuffle and batch the datasets
# train_dataset = train_dataset.shuffle(buffer_size=x_train.getnnz()).batch(batch_size)
# val_dataset = val_dataset.batch(batch_size)
# test_dataset = test_dataset.batch(batch_size)

# print(test_dataset)

"""## **CNN Model**"""

# Reshape data for CNN (assuming a 1D convolution)
x_train_all = x_train.toarray().reshape(x_train.shape[0], x_train.shape[1], 1)
x_test_all = x_test.toarray().reshape(x_test.shape[0], x_test.shape[1], 1)


x_train_all, x_val_all, y_train, y_val = train_test_split(x_train_all, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Define batch size
batch_size = 32

# Define the CNN model
cnn_model = Sequential()
cnn_model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(x_train_all.shape[1], 1)))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Dropout(0.5))
cnn_model.add(Flatten())
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dropout(0.5))
num_classes = len(np.unique(y_train))
cnn_model.add(Dense(num_classes, activation='softmax'))

# Compile the CNN model
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
cnn_model.fit(x_train_all, y_train, epochs=50, batch_size=batch_size, validation_data=(x_val_all, y_val))

# Evaluate the CNN model on the test set
test_loss, test_accuracy = cnn_model.evaluate(x_test_all, y_test)
print(f'CNN Model Test Accuracy: {test_accuracy}')

import matplotlib.pyplot as plt

# Train the CNN model and record history
history = cnn_model.fit(x_train_all, y_train, epochs=50, batch_size=batch_size, validation_data=(x_val_all, y_val))

# Extract training and validation accuracy and loss from history
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Create learning curves
epochs = range(1, len(train_accuracy) + 1)

# Plot training and validation accuracy
plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions
predictions = cnn_model.predict(x_test_all)
predicted_labels = np.argmax(predictions, axis=1)

# Generate confusion matrix
cm = confusion_matrix(y_test, predicted_labels)
print(cm)

# Display the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Generate classification report
report = classification_report(y_test, predicted_labels)
print("Classification Report:\n", report)

# Compute precision, recall, and f1-score for each class
precision_cnn, recall_cnn, f1_cnn, support_cnn = precision_recall_fscore_support(y_test, predicted_labels, average=None)

# Compute weighted average
weighted_precision_cnn = np.sum(precision_cnn * support_cnn) / np.sum(support_cnn)
weighted_recall_cnn = np.sum(recall_cnn * support_cnn) / np.sum(support_cnn)
weighted_f1_cnn = np.sum(f1_cnn * support_cnn) / np.sum(support_cnn)

print(f"Weighted Precision for CNN: {weighted_precision_cnn}")
print(f"Weighted Recall for CNN: {weighted_recall_cnn}")
print(f"Weighted F1-Score for CNN: {weighted_f1_cnn}")

p_cnn = round(weighted_precision_cnn * 100, 2)
r_cnn = round(weighted_recall_cnn * 100, 2)
f1_cnn = round(weighted_f1_cnn * 100, 2)

"""## **LSTM Model**"""

# Define the LSTM model with more hidden layers
lstm_model = Sequential()
lstm_model.add(LSTM(128, input_shape=(x_train_all.shape[1], x_train_all.shape[2]), return_sequences=True))
lstm_model.add(Dropout(0.5))
lstm_model.add(LSTM(64, return_sequences=False))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(32, activation='relu'))
lstm_model.add(Dropout(0.5))
num_classes = len(np.unique(y_train))
lstm_model.add(Dense(num_classes, activation='softmax'))

# Compile the LSTM model
lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the LSTM model
lstm_model.fit(x_train_all, y_train, epochs=10, batch_size=batch_size, validation_data=(x_val_all, y_val))

# Evaluate the LSTM model on the test set
test_loss, test_accuracy = lstm_model.evaluate(x_test_all, y_test)
print(f'LSTM Model Test Accuracy: {test_accuracy}')

# Make predictions
predictions = lstm_model.predict(x_test_all)
predicted_labels = np.argmax(predictions, axis=1)

# Generate confusion matrix
cm = confusion_matrix(y_test, predicted_labels)

# Display the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Generate classification report
report = classification_report(y_test, predicted_labels)
print("Classification Report:\n", report)

"""## **CNN & LSTM Combine**"""

# Define the combined all-LSTM model
combined_model = Sequential()

# all part
combined_model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(x_train_all.shape[1], 1)))
combined_model.add(MaxPooling1D(pool_size=2))
combined_model.add(Dropout(0.5))
combined_model.add(Flatten())

# Reshape output for LSTM
combined_model.add(Reshape((combined_model.output_shape[1], 1)))

# LSTM part
combined_model.add(LSTM(128, return_sequences=True))

# Additional LSTM layer
combined_model.add(LSTM(64, return_sequences=False))

# Fully connected layers
combined_model.add(Dense(64, activation='relu'))
combined_model.add(Dropout(0.5))

# Additional hidden layer
combined_model.add(Dense(32, activation='relu'))

num_classes = len(np.unique(y_train))
combined_model.add(Dense(num_classes, activation='softmax'))

# Compile the model
combined_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
combined_model.fit(x_train_all, y_train, epochs=10, batch_size=batch_size, validation_data=(x_val_all, y_val))

# Evaluate the model on the test set
test_loss, test_accuracy = combined_model.evaluate(x_test_all, y_test)
print(f'Combined Model Test Accuracy: {test_accuracy}')

"""## **Stack LSTM**"""

# Define the stacked LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(128, return_sequences=True, input_shape=(1, x_train_all.shape[2])))
lstm_model.add(Dropout(0.5))
lstm_model.add(LSTM(64, return_sequences=False))
lstm_model.add(Dropout(0.5))
num_classes = len(np.unique(y_train))
lstm_model.add(Dense(num_classes, activation='softmax'))

# Compile the LSTM model
lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the LSTM model
lstm_model.fit(x_train_all, y_train, epochs=10, batch_size=batch_size, validation_data=(x_val_all, y_val))

# Evaluate the LSTM model on the test set
test_loss, test_accuracy = lstm_model.evaluate(x_test_all, y_test)
print(f'Stacked LSTM Model Test Accuracy: {test_accuracy}')

"""## **Convolution 1D**"""

# Define the CNN model
model = Sequential()

# Embedding layer
model.add(Embedding(input_dim=x_train.shape[1], output_dim=100, input_length=x_train.shape[1]))

# 1D Convolutional layer
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(2))
model.add(GlobalMaxPooling1D())

# Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(len(set(y_train)), activation='softmax'))  # Number of units equal to the number of classes

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test data
y_pred = model.predict(x_test)
y_pred_classes = [tf.argmax(pred) for pred in y_pred]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
print("Test Accuracy:", accuracy)

"""# **MLP Code**"""

# https://colab.research.google.com/drive/1h83DaVAjwVaf5suph-Dt42Ie3UGvMNPf?authuser=1#scrollTo=hYLyCxpV4Wse

"""# **Precision Comparison**"""

p_mlp = round(0.9393939393939393 * 100, 2)

# Algorithm names
algorithms = ['Decision Tree', 'SVM', 'KNN', 'NB', 'LR', 'CNN', 'MLP']

# Precision values
precision_values = [p_dec, p_svc, p_knn, p_nb, p_lr, p_cnn, p_mlp]

# Create a vertical bar graph
plt.bar(algorithms, precision_values, edgecolor='none')
plt.title('Comparison of Precision Values')
plt.ylim(0, 100)  # Set the y-axis limit to 0-100 for percentage values

# Display the precision values on top of the bars
for i, value in enumerate(precision_values):
    plt.text(i, value + 1, f'{value:.2f}', ha='center', va='bottom')
plt.yticks([])
# Show the plot
plt.show()

"""# **Recall Comparison**"""

r_mlp = round(0.9090909090909091 * 100, 2)

# Algorithm names
algorithms = ['Decision Tree', 'SVM', 'KNN', 'NB', 'LR', 'CNN', 'MLP']

# Recall values
recall_values = [r_dec, r_svc, r_knn, r_nb, r_lr, r_cnn, r_mlp]

# Create a horizontal bar graph
plt.barh(algorithms, recall_values, edgecolor='none', color='skyblue')
plt.title('Comparison of Recall Values')
plt.xlim(0, 100)  # Set the x-axis limit to 0-100 for percentage values

# Display the recall values on the right of the bars
for i, value in enumerate(recall_values):
    plt.text(value + 1, i, f'{value:.2f}', ha='left', va='center')
plt.xticks([])
# Show the plot
plt.show()

"""# **F1 Score Comparison**"""

f1_mlp = round(0.9026973026973028 * 100, 2)

# Algorithm names
algorithms = ['Decision Tree', 'SVM', 'KNN', 'NB', 'LR', 'CNN', 'MLP']

# f1Score values
f1Score_values = [f1_dec, f1_svc, f1_knn, f1_nb, f1_lr, f1_cnn, f1_mlp]

# Create a horizontal bar graph
plt.barh(algorithms, f1Score_values, edgecolor='none', color='blue')
plt.title('Comparison of F1 Score Values')
plt.xlim(0, 100)  # Set the x-axis limit to 0-100 for percentage values

# Display the f1Score values on the right of the bars
for i, value in enumerate(f1Score_values):
    plt.text(value + 1, i, f'{value:.2f}', ha='left', va='center')
plt.xticks([])
# Show the plot
plt.show()