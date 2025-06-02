
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

##LOAD LSTM
model=load_model("next_word_lstm.h5")

##LOAD tokenizer
with open('tokenizer.pickle','rb') as handle:
  tokenizer=pickle.load(handle)

##predictor fnc
def predict_next_word(model, tokenizer, text, max_sequence_len):
  token_list=tokenizer.texts_to_sequences([text])[0]
  if len(token_list) >= max_sequence_len:
    token_list=token_list[-(max_sequence_len-1):] ##nsring length matchs max_seqquence
  token_list=pad_sequences([token_list],maxlen=max_sequence_len-1, padding='pre')
  predicted=model.predict(token_list,verbose=0)
  predicted_word_index=np.argmax(predicted, axis=1)
  for word, index in tokenizer.word_index.items():
    if index == predicted_word_index:
      return word
  return None

  ##sstreamlit app

  st.title("Next word prediction")
  inpt_text=st.text_input("Enter the sequence of words",'to be or not to be')

  if st.button('Predict next word'):
    max_sequence_len = model.input_shape[1]+1  ##retrieving the max_length from model
    next_word=predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f"Next word :{next_word}")
    
