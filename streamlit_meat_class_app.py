import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


st.set_page_config(page_title="Meat Classifier", page_icon="🥩", layout="wide")

st.markdown("""<h1 class="title">Spoiled or Fresh Meat? 🥩</h1>""", unsafe_allow_html=True)
st.markdown("""<h3 class="subtitle">A CNN that classifies meat as spoiled or fresh</h3>""", unsafe_allow_html=True)

# Spacing
def spacing(num_spaces):
    for i in range(num_spaces):
        st.header("")


spacing(5)

fresh_col, spoiled_col = st.columns(2)

with fresh_col:
    st.markdown("""<h1 class="fresh">Fresh Meat Example</h1>""", unsafe_allow_html=True)

    st.image("Meat Classifier/Fresh/test_20171016_175521D.jpg")
    st.button("Show Fresh Picture", key="fresh")

with spoiled_col:

    st.markdown("""<h1 class="spoiled">Spoiled Meat Example</h1>""", unsafe_allow_html=True)

    st.image("Meat Classifier/Spoiled/test_20171017_233921D.jpg")
    st.button("Show Spoiled Picture", key="spoiled")




spacing(8)

st.header("My Profiles")

#Profiles to Kaggle and GitHub
st.markdown("""<ul class="profiles">
            <li><a href="">Kaggle</a></l1>
            <li>GitHub</l1>
            </ul>
""", unsafe_allow_html=True)

spacing(2)

st.markdown("**Kaggle Citation**")
st.write("""@inproceedings{ulucan2019meat,
title={Meat quality assessment based on deep learning},
author={Ulucan, Oguzhan and Karakaya, Diclehan and Turkan, Mehmet},
booktitle={2019 Innovations in Intelligent Systems and Applications Conference (ASYU)},
pages={1--5},
year={2019},
organization={IEEE}
}""")

st.write("O.Ulucan , D.Karakaya and M.Turkan.(2019) Meat quality assessment based on deep learning.In Conf Innovations Intell. Syst. Appli. (ASYU)")

st.write("Link to Kaggle dataset: https://www.kaggle.com/datasets/crowww/meat-quality-assessment-based-on-deep-learning")

# css styling for elements above
st.markdown("""

    <style>

    *{
        background-color: MintCream;
    }

    .title{
        font-size : 80px;
        color: #006600;
        text-align: center;
    }

    .subtitle{
        text-align: center;
    }

    .fresh{
        color: Crimson;
        text-align:center;
    }  

    .spoiled{
        color: SaddleBrown;
        text-align: center;
    } 
    
    </style>
""", unsafe_allow_html=True)
