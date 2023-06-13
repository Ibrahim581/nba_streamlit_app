import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image


st.write("""
# Make or Miss?
#### By: Shaikh Ibrahim Noman
###### Link to data set in GitHub
###### Welcome to make or miss. This web app predicts whether a basketball shot will go in or not. Select the parameters on the lfet hand side and see what happens.
""")

image = Image.open('basketball.png')
st.image(image, width = 300)

def user_input_features():
    DRIBBLES = st.sidebar.slider('Dribbles', 0, 5, 28)
    SHOT_DIST = st.sidebar.slider('Shot Distance', .1, 4.4, 45.3)
    CLOSE_DEF_DIST = st.sidebar.slider('Defender Distance', .1, 6.9, 53.2)
    TOUCH_TIME = st.sidebar.slider('Touch Time', .1, 2.5, 23.3)
    data = {'DRIBBLES': DRIBBLES,
            'SHOT_DIST': SHOT_DIST,
            'CLOSE_DEF_DIST': CLOSE_DEF_DIST,
            'TOUCH_TIME': TOUCH_TIME}
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

st.subheader('User Parameters')
st.write(df)

data = pd.read_csv('data_filtered.csv')
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')

label_encoder = LabelEncoder()

X_train = X_train[['DRIBBLES', 'SHOT_DIST', 'CLOSE_DEF_DIST', 'TOUCH_TIME']]
y_train_selected = y_train['SHOT_RESULT']
y_train_encoded = label_encoder.fit_transform(y_train_selected)

if st.button('Submit'):
    model_rf = RandomForestClassifier(criterion = 'entropy', n_estimators = 90, max_features = 2,min_samples_split = 6, random_state = 7)
    model_rf = model_rf.fit(X_train, y_train_encoded)
    predictions = model_rf.predict(df)
    prediction_proba = model_rf.predict_proba(df)

    st.subheader('Prediction')
    st.write(data.SHOT_RESULT[predictions])
    #st.write(prediction)

    st.subheader('Prediction Probability')
    st.write(prediction_proba)
