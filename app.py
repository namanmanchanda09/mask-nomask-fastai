import streamlit as st
from fastai.vision.all import *

st.set_page_config(
    page_title="Prediction App",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

learn = load_learner('model-2.pkl')

def main():
    st.title('Face mask detection')
    st.write('Project by:')
    st.write("CSE - I - 46")
    st.write("Malika Singh")
    st.write("Yogya Pankaj Mendiratta")
    st.write("Apoorv Minocha")
    image_file = st.file_uploader('Upload a image file', ['png','jpg','jpeg'], accept_multiple_files=False)
    if image_file is not None:
        file_details = {"FileName":image_file.name,"FileType":image_file.type}
        # st.write(file_details)
        image = load_image(image_file)

        img = PILImage.create(image_file)
        # is_cat, _, probs = learn.predict(img)

        pred,pred_idx,probs = learn.predict(img)
        # print(f"Is this a cat?: {is_cat}.")
        # print(f"Probability it's a cat: {probs[1].item():.6f}")
        # st.write(f"Is this a cat?: {is_cat}.")
        # st.write(f"Probability it's a cat: {probs[1].item():.6f}")

        st.write(f"Prediction: {pred}; Probability: {probs[pred_idx]:.04f}")
        


        st.image(image)
    
if __name__ == main():
    main()
