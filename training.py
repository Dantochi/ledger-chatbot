import random, pickle, json, nltk
import numpy as np
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

from tensorflow import keras

word_collapse = WordNetLemmatizer()  # Collapsing of similar words to one
intents = json.loads(open('intent.json').read())
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intent']:
    for inputs in intent['input']:
        word_list = nltk.word_tokenize(inputs)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
# print(documents)
# print(words)
words = [word_collapse.lemmatize(str(word)) for word in words if word not in ignore_letters]
words = sorted(
    set(words))  # The set takes out the duplicates but the sorted sorts the element in ascending or descending order

classes = sorted(set(classes))
pickle.dump(words, open('words.pkl',
                        'wb'))  # This pickle module helps in serialization or conversion of a python object such as
# list or arrays into a format that can be easily stored by the computer. As we all know it's the binary format.
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
bag_list = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [word_collapse.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)  # So I believe this line is trying to get the frequency of appearance of the words in the array word_patterns
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])
    bag_list.append([bag])

random.shuffle(training)
random.shuffle(bag_list)
print(training)
training = np.array(training)
bag_list = np.array(bag_list)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = keras.models.Sequential()
model.add(keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(len(train_y[0]), activation='softmax'))

sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.keras', hist)
print('Done')
