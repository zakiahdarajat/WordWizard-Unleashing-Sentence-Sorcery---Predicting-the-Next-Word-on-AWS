from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import os
import logging

from .sustain import tokenizer

logger = logging.getLogger(__name__)

# Create your views here.
def handler(request):
    response = "Input the word first"
    if request.method == 'POST':
        text = request.POST.get('Name', '').strip()
        if text:
            try:
                response = predicting(text)
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                response = "Error occurred during prediction. Please try again."
        else:
            response = "Please enter some text to predict."
    return render(request, "index.html", {'response': response})


def predicting(data):
    def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
        result = list()
        in_text = seed_text
        # generate a fixed number of words
        for _ in range(n_words):
            # encode the text as integer
            encoded = tokenizer.texts_to_sequences([in_text])[0]
            # truncate sequences to a fixed length
            encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
            # predict probabilities for each word
            predict_x = model.predict(encoded, verbose=0)
            yhat = np.argmax(predict_x, axis=1)[0]  # Fix: extract scalar from array
            # map predicted word index to word
            out_word = ''
            for word, index in tokenizer.word_index.items():
                if index == yhat:  # Now comparing scalar to scalar
                    out_word = word
                    break
            
            # If no word found, break to avoid infinite loop
            if not out_word:
                logger.warning(f"No word found for index {yhat}")
                break
                
            # append to input
            in_text += ' ' + out_word
            result.append(out_word)
        return ' '.join(result)

    # select a seed text
    seed_text = data
    print(f"Input: {seed_text}")
    seq_length = 50
    res_length = 12 # predict next 12 words.
    
    try:
        # load the model with absolute path
        model_path = os.path.join(settings.BASE_DIR, 'polls', 'nextWord.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        model = load_model(model_path)
        Tokenizer = tokenizer
        
        if Tokenizer is None:
            raise ValueError("Tokenizer not loaded properly")
            
        # generate new text
        generated = generate_seq(model, Tokenizer, seq_length, seed_text, res_length)
        return generated
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return f"Prediction failed: {str(e)}"