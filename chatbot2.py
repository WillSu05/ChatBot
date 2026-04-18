import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

# Asegúrate de que estos nombres coincidan con tus archivos
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5') # o .keras

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
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
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    # VALIDACIÓN PARA EVITAR EL ERROR DE "OUT OF RANGE"
    if not intents_list:
        return "Lo siento, no entiendo eso. ¿Puedes preguntar de otra forma?"
    
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            # CORCHETES, NO PARÉNTESIS
            result = random.choice(i['responses'])
            break
    return result

print("¡Bot listo! Escribe algo para comenzar (escribe 'salir' para terminar):")

# EL BUCLE PRINCIPAL DEBE SER ASÍ:
while True:
    message = input("Tú: ") # Agregamos el "Tú: " para que sepas dónde escribir
    if message.lower() == "salir":
        break
    
    # 1. Predecir la clase
    ints = predict_class(message)
    # 2. Obtener la respuesta
    res = get_response(ints, intents)
    # 3. Imprimir
    print("Bot:", res)