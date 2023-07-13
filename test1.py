# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split 
# from sklearn.svm import SVC


# df = pd.read_csv('iris-species.csv')
# df['Label'] = df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# x = df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
# y = df['Label']

# xtrain,xtest,ytrain,ytest = train_test_split(x,y, test_size = 0.33, random_state = 42)

# svc = SVC(kernel = 'linear')
# svc.fit(xtrain,ytrain)
# score = svc.score(xtrain,ytrain)


# @st.cache()
# def prediction(SepalLength, SepalWidth, PetalLength, PetalWidth):
#   species = svc.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
#   species = species[0]
#   if species == 0:
#   	return 'Iris-setosa'
#   elif species == 1:
#   	return 'Iris-virginica'
#   else:
#   	return 'Iris-versicolor'

# st.title('Iris Flower App')

# s_length = st.slider('SepalLength',0.0,10.0)
# p_length = st.slider('PetalLength',0.0,10.0)
# s_width = st.slider('SepalWidth',0.0,10.0)
# p_width = st.slider('PetalWidth',0.0,10.0)


# if st.button('Predict'):
#   species_type = prediction(s_length,s_width,p_length,p_width)
#   st.write("Species Predicted: ",species_type)
#   st.write("Accuracy score:",score)



############------------new

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split as tts
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('iris-species.csv')
df['Label'] = df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

x = df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Label']

xtrain,xtest,ytrain,ytest = tts(x,y,test_size=0.33,random_state=42)

svc= SVC(kernel ='linear')
svc.fit(xtrain,ytrain)

log = LogisticRegression(n_jobs=-1)
log.fit(xtrain,ytrain)

rf = RandomForestClassifier(n_jobs=-1,n_estimators=100)
rf.fit(xtrain,ytrain)




@st.cache()
def prediction(model,sl,sw,pl,pw):
  species = model.predict([[sl,sw,pl,pw]])
  if species[0] == 0:
    return 'Iris-setosa'
  elif species[0] == 1:
    return 'Iris-virginica'
  else:
    return 'Iris-versicolor'
    
st.title('Iris Flower App')

st.sidebar.title('Detect Flower')

sl = st.sidebar.slider('SepalLengthCm',float(df['SepalLengthCm'].min()),float(df['SepalLengthCm'].max()))
sw = st.sidebar.slider('SepalWidthCm',float(df['SepalWidthCm'].min()),float(df['SepalWidthCm'].max()))
pl = st.sidebar.slider('PetalLengthCm',float(df['PetalLengthCm'].min()),float(df['PetalLengthCm'].max()))
pw = st.sidebar.slider('PetalWidthCm',float(df['PetalWidthCm'].min()),float(df['PetalWidthCm'].max()))

choice = st.sidebar.selectbox('Classifier',('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))

if st.sidebar.button('Predict'):
  if choice == 'Support Vector Machine':
    flower = prediction(svc,sl,sw,pl,pw)
    score = svc.score(xtrain,ytrain)

  if choice == 'Logistic Regression':
    flower = prediction(log,sl,sw,pl,pw)
    score = log.score(xtrain,ytrain)

  if choice == 'Random Forest Classifier':
    flower = prediction(rf,sl,sw,pl,pw)
    score = rf.score(xtrain,ytrain)

  st.write('Flower Species is ',flower)
  st.write('Accuracy of the model ',score)
