import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import pandas as pd 
import webbrowser
from PIL import Image, ImageOps
import random

cap = cv2.VideoCapture(0)

def cal_confi(State):
  global event_sum
  data = pd.read_csv("weights_data/Data.csv")                           
  df = pd.DataFrame(data)
  loc_state = df.loc[df['State'] == State] 
  event = np.array(loc_state)
  event_sum = (event[0][1]+event[0][2])

def calc_result(score):
   global total_score
   if result == "Crack":
       crack_score = random.randint(10, 25)
       total_score = crack_score + score
       if st.button('View Results'):
            if result == "Crack":
                st.error("Crack Detected")
                st.error(f"The life expectancy of this structure in {State} is {total_score}%")
                if total_score in range (25):
                                      st.error("Plastic shrinkage concrete cracks, and needs to be corrected as soon as possible. Also, the cost of this crack's type is CAD 500")
                if total_score in range (25,35):
                                      st.error("Expansion concrete cracks, and needs to be corrected as soon as possible. Also, the cost of this crack's type is CAD 900")
                if total_score in range (35,45):
                                      st.error("Settling concrete cracks, and needs to be corrected as soon as possible. Also, the cost of this crack's type is CAD 1300")
                if total_score in range (45,55):
                                      st.error("Heaving concrete cracks, and needs to be corrected as soon as possible. Also, the cost of this crack's type is CAD 2200")
                if total_score in range (55,65):
                                      st.error("Concrete cracks caused by premature drying, and needs to be corrected as soon as possible. Also, the cost of this crack's type is CAD 3000")
                if total_score in range (65,100):
                                      st.error("Concrete cracks caused by overloading the slab, and needs to be corrected as soon as possible. Also, the cost of this crack's type is CAD 4300")

   elif result == "Not-Crack":
       crack_score = random.randint(25, 40)
       total_score = crack_score + score
       if st.button('View Results'):
            st.success("No Crack Detected")
        #    st.success(f"The life expectancy of this structure in {State} is {total_score}%")
            st.success("There is no such harmful crack present, and the structure is strong enough physically.")


st.markdown("""
<style>
.medium-button {
   color:white
}
body {
  color: black;
 
}
</style>
    """, unsafe_allow_html=True)
page_bg_img = '''
<style>
body {
background-image: url("https://images.pexels.com/photos/2098428/pexels-photo-2098428.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=750&w=1260");
background-size: cover;
}
.markdown-text-container{
  color:white !important;
}
.stMarkdown{
  color:white;
}
.stFileUploader label{
 color:white;
}
.stSelectbox label,.uploadedFileName{
 color:white; 
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("Welcome to the Structural Defect Detection Program!")
st.header("Please fill the following details properly:")
State = st.selectbox(
    'Select State Name',
    ('Ontario', 'Quebec', 'Nova Scotia', 'New Brunswick', 'Manitoba', 'British Columbia', 'Prince Edward Island', 'Saskatchewan', 'Alberta', 'Newfoundland and Labrador')
)

file = st.file_uploader("Please upload your image here:")


 
def upload_and_predict(image_data,weight):

  global prediction
  size = (224,224)
  image = ImageOps.fit(image_data,size,Image.ANTIALIAS)
  img  = np.array(image)
  img_reshape = img[np.newaxis,...]
  prediction = weight.predict(img_reshape)
  return prediction
  
  st.sidebar.markdown("---")
    
  st.subheader("Output")
  stframe = st.empty()
    
  run = st.sidebar.button("Start")
  stop = st.sidebar.button("Stop")
  st.sidebar.markdown("---")
    
  cv2.imshow("Webcam", frame)
  if cv2.waitKey(1) == ord('q'):
     break

  
if file is  None:
   st.error("Please upload an image")
else:   
   try:
     image = Image.open(file)
     rebuild_model = load_model("weights_data/my_model.h5")
     rebuild_model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])
     upload_and_predict(image,rebuild_model)
     class_name = ['Crack','Not-Crack']
     result = class_name[np.argmax(prediction)]
     cal_confi(State)
     calc_result(event_sum)
   except ValueError:
      st.warning("Please upload a valid image")

