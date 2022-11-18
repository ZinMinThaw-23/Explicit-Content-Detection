import streamlit as st
import pandas as pd
import re
import os
import pickle
from keras.models import load_model
from DataPreprocessing import CustomFeats
 

#Streamlit Theme
st.set_page_config( page_title="Color Analysis with David",page_icon="🧊",layout="wide", initial_sidebar_state="expanded")      

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://cdn.pixabay.com/photo/2017/08/07/21/53/still-2608285_1280.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()   

streamlit_style = """
			<style>
			@import url("https://fonts.shopifycdn.com/nunito_sans/nunitosans_n7.5bd4fb9346d13afb61b3d78f8a1e9f31b128b3d9.woff2?h1=c3RvcmUudGF5bG9yc3dpZnQuY29t&hmac=60eccdbd2670eec353d0081774bf120b82d525923d9ffdd1d1f37aaf86d8bcb8") format("woff2"), url("https://fonts.shopifycdn.com/nunito_sans/nunitosans_n7.2bcf0f11aa6af91c784a857ef004bcca8c2d324d.woff?h1=c3RvcmUudGF5bG9yc3dpZnQuY29t&hmac=2bc2b91b35c84d85452727bfe0d6cc4ac43bf8978e2a27cfb29c154128a4a4cc") format("woff");
			html, body, [class*="css"]  {
			font-family: 'Nunito Sans', sans-serif;
			font-size:15px;
			color: 	#ffffff;
			}
			</style>
			"""
st.markdown(streamlit_style, unsafe_allow_html=True)

st.markdown("""
<style>
.big-font {
    font-size:100px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p style="font-size: 42px;">Explicit Content Analysis in Music Lyrics </p>', unsafe_allow_html=True)

form = st.form(key='my-form')
input = form.text_input("Input :","I was Enchanted to meet you")
submit = form.form_submit_button('Submit')
st.write('Press submit to apply')

def classification(feats, model):
  test_vecs = feats.transform(test_data)
    
  model.fit(train_vecs, train_label)

  train_preds = model.predict(train_vecs)

  return train_preds
	
while submit:
	#vocab_size = 500
	#embedding_dim = 64
	#max_length = 16067
	#trunc_type='post'        #put needed '0's for max length
	#oov_tok = "<OOV>"
	
	#tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
	#tokenizer.fit_on_texts(input)
	#word_index = tokenizer.word_index
	#sequences = tokenizer.texts_to_sequences(input)
	#padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type,padding='post')
	
	
	feats = pickle.load(open("feats.sav", 'rb'))
	input_vecs = feats.transform(input)
	loaded_model = pickle.load(open("finalized_model_rf.sav", 'rb'))
	output = loaded_model.predict(input_vecs)


#if output[0]==1:
	#output_string="The song includes explicit words"
#else:
	#output_string="The song is clean and doesn't include explicit words"
#st.text_area(label="Output Data:", value=output_string, height=50)
st.text_area(label="Output Data:", value=output, height=50)

#st.text_area(label="Output Data:", value=output_string, height=50)
