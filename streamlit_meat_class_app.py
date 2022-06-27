import numpy as np
import tensorflow as tf
import streamlit as st

st.set_page_config(page_title="Meat Classifier", page_icon="🥩", layout="wide")
MODEL = tf.keras.models.load_model("meat_classifier_transfer_cnn_model.h5")

def spacing(num_spaces):
    for i in range(num_spaces):
        st.header("")

def format_uploaded_files(raw_image):
    
    img = tf.keras.utils.load_img(raw_image)
    img = tf.keras.utils.img_to_array(img)

    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    
    img = np.array(tf.image.resize(img, [224, 224]))
    img = img.reshape(1, 224, 224, 3)

    return img

def make_predictions(files_uploaded):
    
    formatted_images = list(map(format_uploaded_files, files_uploaded))

    preds = []

    for pic in formatted_images:
        preds.append(MODEL.predict(pic))

    preds = [0 if pred < .5 else 1 for pred in preds]
    return preds


def write_prediction(pred_col, files_uploaded):

    with pred_col:

        # If there is only 1 file uploaded, show it.
        if len(files_uploaded) == 1:
            st.image(files_uploaded)

        image_predictions = make_predictions(files_uploaded)

        for pred in image_predictions:

            if pred == 0:
                st.markdown("""<h2 class="centerclass">Fresh ✅</h2>""", unsafe_allow_html=True)
            else:
                st.markdown("""<h2 class="centerclass">Spoiled ❌</h2>""", unsafe_allow_html=True)


        


def citation():
    st.markdown("**Kaggle Citation**")
    st.write("""@inproceedings{ulucan2019meat,
    title={Meat quality assessment based on deep learning},
    author={Ulucan, Oguzhan and Karakaya, Diclehan and Turkan, Mehmet},
    booktitle={2019 Innovations in Intelligent Systems and Applications Conference (ASYU)},
    ages={1--5},
    year={2019},
    organization={IEEE}
    }""")

    st.write("O.Ulucan , D.Karakaya and M.Turkan.(2019) Meat quality assessment based on deep learning.In Conf Innovations Intell. Syst. Appli. (ASYU)")
    st.write("Link to Kaggle dataset: https://www.kaggle.com/datasets/crowww/meat-quality-assessment-based-on-deep-learning")


st.markdown("""<h1 class="title">Spoiled or Fresh Meat? 🥩</h1>""", unsafe_allow_html=True)
st.markdown("""<h3 class="centerclass">A CNN that classifies meat as spoiled or fresh</h3>""", unsafe_allow_html=True)


spacing(5)

fresh_col, spoiled_col = st.columns(2)

with fresh_col:
    st.markdown("""<h1 class="fresh">Fresh Meat Example</h1>""", unsafe_allow_html=True)

    fresh_img_expander = st.expander("Toggle Fresh Image", expanded=True)

    with fresh_img_expander:
        st.image("test_20171016_175521D.jpg")
    
with spoiled_col:

    st.markdown("""<h1 class="spoiled">Spoiled Meat Example</h1>""", unsafe_allow_html=True)

    spoiled_img_expander = st.expander("Toggle Spoiled Image", expanded=True)

    with spoiled_img_expander:
        st.image("test_20171018_233921D.jpg")

spacing(4)

file_upload_col, pred_col = st.columns(2)

with file_upload_col:
    st.header("Upload photo of meat sample from data directory")
    files_uploaded = st.file_uploader("", accept_multiple_files=True, type=["png", "jpg", "jpeg"], key="file")

with pred_col:
    st.markdown("""<h2 class="centerclass">Prediction</h2>""", unsafe_allow_html=True)

pred_button = st.button("Make Prediction", key="pred_button")

if pred_button and files_uploaded:
    write_prediction(pred_col, files_uploaded)

elif pred_button and not files_uploaded:
    with pred_col:
        st.markdown("""<h2 class="no_file">Please choose a file</h2>""", unsafe_allow_html=True)


spacing(5)

heatmap_col, classification_report_col = st.columns(2)

with heatmap_col:
    st.markdown("""<h1 class="centerclass">Confusion Matrix</h1>""", unsafe_allow_html=True)
    st.image("heatmap.png")

with classification_report_col:
    st.markdown("""<h1 class="centerclass">Classification Report Stats</h1>""", unsafe_allow_html=True)
    st.image("class_report.png")

spacing(5)

st.title("My Profile Links")

#Profiles to Kaggle and GitHub
st.markdown("""<ul class="profiles">
            <li class="kaggle"><a href="https://www.kaggle.com/mattkerekes/code">Kaggle</a></l1>
            <li class="github"><a href="https://github.com/mkerekes1427/MeatClassifier">GitHub</a></l1>
            </ul>
""", unsafe_allow_html=True)

spacing(2)

citation()

# css styling for elements above
st.markdown("""

    <style>

    *{
        background-color: MintCream;
    }

    .title{
        font-size : 80px;
        color: DarkSlateGrey;
        text-align: center;
    }

    .centerclass{
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

    .no_file{
        color: FireBrick;
        text-align: center;
    } 


    .profiles li{
        display: inline;
    }

    .kaggle a{
        font-size: 36px;
        color: Teal;
        text-decoration: None;
    }

    .github a{
        font-size: 36px;
        color: RebeccaPurple;
        text-decoration: None;
    }
    

    </style>
""", unsafe_allow_html=True)
