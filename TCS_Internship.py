#!/usr/bin/env python
# coding: utf-8

# # Import necessary libraries

# In[31]:


import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# In[32]:


import nltk
nltk.download('punkt')


# # Load and preprocess the data

# In[33]:


data = pd.read_csv(r'D:\Machine Learning\Internship\IMDB Dataset.csv\IMDB Dataset.csv')


# In[34]:


pd.set_option('display.max_colwidth', None)
data.head()


# In[35]:


data.shape


# In[36]:



data.info()


# In[37]:


data.describe()


# In[44]:


data.isna().sum()


# In[45]:


text = data['review'].values
labels = data['sentiment'].values


# In[46]:


# Encode labels as integers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)


# In[47]:


tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(text)


# In[48]:


sequences = tokenizer.texts_to_sequences(text)
padded_sequences = pad_sequences(sequences, maxlen=100)
labels = to_categorical(labels)


# In[49]:


# Split the dataset into training and test sets
train_indices = np.random.choice(len(padded_sequences), int(len(padded_sequences)*0.9), replace=False)
test_indices = np.array(list(set(range(len(padded_sequences))) - set(train_indices)))
train_data = padded_sequences[train_indices]
train_labels = labels[train_indices]
test_data = padded_sequences[test_indices]
test_labels = labels[test_indices]


# # Training the Model

# In[50]:


# Define the model architecture
model = Sequential()
model.add(Embedding(10000, 128, input_length=100))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(32, activation='relu', dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))


# In[51]:


# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[52]:


model.summary()


# In[53]:


# Train the model
history = model.fit(padded_sequences, labels, epochs=10, batch_size=32, validation_split=0.1)


# In[54]:


# Plot the accuracy history
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[55]:


# Test the model on new data
test_text = ["This is a great movie", "I hate this food", "The service was terrible"]
test_sequences = tokenizer.texts_to_sequences(test_text)
padded_test_sequences = pad_sequences(test_sequences, maxlen=100)
predictions = model.predict(padded_test_sequences)


# In[56]:


# Print the predicted sentiment labels
print(predictions.argmax(axis=1))


# In[57]:


paragraph = "Climate change is a highly controversial and complex topic that has sparked intense debate among scientists, politicians, and the public at large. Some argue that the evidence for human-caused climate change is overwhelming, and that urgent action is needed to mitigate its effects. Others contend that the science is far from settled, and that natural factors such as sunspots and volcanic activity could be responsible for the observed changes in the Earth's climate.One thing that is clear is that the Earth's climate has changed dramatically over time, long before humans appeared on the scene. Ice ages have come and gone, sea levels have risen and fallen, and the planet has experienced periods of extreme heat and cold. Some argue that the current warming trend is simply part of a natural cycle, and that it will eventually reverse itself without human intervention.However, the overwhelming majority of scientific evidence suggests that the current warming trend is different from anything the Earth has experienced in the past. The concentration of greenhouse gases in the atmosphere has increased dramatically since the Industrial Revolution, largely as a result of human activity such as the burning of fossil fuels. This increase in greenhouse gases is causing the Earth's temperature to rise, leading to melting ice caps, rising sea levels, and more frequent and intense heatwaves, storms, and other weather events.Despite the scientific consensus on the reality and seriousness of climate change, there are still many skeptics who refuse to accept the evidence. Some argue that the data has been manipulated, or that scientists are simply pushing a political agenda."


# In[58]:


# Tokenize the paragraph into sentences
sentences = nltk.sent_tokenize(paragraph)

# Convert sentences to sequences
sequences = tokenizer.texts_to_sequences(sentences)

# Pad sequences
padded_sequences = pad_sequences(sequences, maxlen=100)

# Make predictions
predictions = model.predict(padded_sequences)

# Get the sentiment labels for each sentence
sentiment_labels = predictions.argmax(axis=1)

# Count the number of positive and negative sentences
num_positive = np.count_nonzero(sentiment_labels == 0)
num_negative = np.count_nonzero(sentiment_labels == 1)


# In[59]:


# Print the number of positive and negative sentences
print(f"Number of positive sentences: {num_positive}")
print(f"Number of negative sentences: {num_negative}")


# In[60]:


# Determine the overall sentiment of the paragraph
if num_positive > num_negative:
    overall_sentiment = "positive"
elif num_positive == num_negative:
    overall_sentiment = "neutral"
else:
    overall_sentiment = "negative"

# Print the overall sentiment
print(f"Overall sentiment: {overall_sentiment}")


# In[61]:


# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test accuracy:', test_acc)


# In[62]:


# Create a bar chart of the number of positive and negative sentences
fig, ax = plt.subplots()
ax.bar(['Positive', 'Negative'], [num_positive, num_negative])
ax.set_xlabel('Sentiment')
ax.set_ylabel('Number of Sentences')
ax.set_title('Sentiment Analysis Results')
plt.show()


# In[63]:


# Create a pie chart of the proportion of positive and negative sentences
fig, ax = plt.subplots()
ax.pie([num_positive, num_negative], labels=['Positive', 'Negative'], autopct='%1.1f%%')
ax.set_title('Sentiment Analysis Results')
plt.show()


# In[ ]:




