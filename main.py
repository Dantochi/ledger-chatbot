import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import random
import pickle
import numpy as np
import json  # This is required to handle/load the intents created in the intents.json folder

words_collapse = WordNetLemmatizer()  # This is used to collapse similar words to one word such as builds, built,
# building to build
# This json.loads() used to convert the json file into format that is readable, editable and accessible by python.
# From JSON to dictionary but structure is maintained.
intents = json.loads(open("intent.json").read())
words = pickle.load(open('words.pkl', 'rb'))  # rb stands for read bytes or read binary
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.keras')


def clean_up_sentence(sentence):  # This function returns the list of words in a particular text without any duplicates
    # This nltk.word_tokenize() helps to split words, its like the machine reading a text word for word and the
    # result is used to find trends in text
    sentence_words = nltk.word_tokenize(sentence)
    # Using the lemmatize methods help to flush all duplicates of words so that only one of each word remains.
    sentence_words = [words_collapse.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    rest = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(rest) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    global result
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intent']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


print("GO! Bot is running!")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
