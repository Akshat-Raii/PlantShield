import streamlit as st
import tensorflow as tf
import numpy as np

st.set_page_config(
    page_title="PlantShield",
    page_icon="Images\plant (1).png",  # Make sure the path is correct
)

# Tensorflow Model Prediction
def model_prediction(test_image):
    model=tf.keras.models.load_model("already_trained_model.keras")
    image=tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])   # Convert Single Image to a batch
    prediction=model.predict(input_arr)
    result_index=np.argmax(prediction)
    return result_index

# Sidebar
st.sidebar.title("Dashboard")
app_mode=st.sidebar.selectbox("Select Page",["Home","About","Disease Detector"])


#Home page
if(app_mode=="Home"):
    st.header("PlantShield: Your Online Plant Disease Detection Hub")
    image_path="Images\Patch plants.gif"
    st.image(image_path,use_column_width=True) # Stretch it to whole page
    st.markdown("""
    ## Welcome to the PlantShield System! 🌱🔍

    Welcome to PlantShield, where we help you quickly and accurately identify plant diseases. Just upload an image of your plant, and our system will analyze it to detect any signs of disease. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Visit the **Disease Detector** page and upload an image of your plant showing symptoms.
    2. **Analysis:** Our advanced algorithms will process the image to identify any potential diseases.
    3. **Results:** View the results and receive recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** We use cutting-edge machine learning techniques for precise disease detection.
    - **User-Friendly:** Our intuitive interface ensures a seamless experience.
    - **Fast and Efficient:** Get results in seconds, enabling quick decision-making.

    ### Get Started
    Click on the **Disease Detector** page in the sidebar to upload an image and experience the power of PlantShield!

    ### About Us
    Learn more about our project and goals on the **About** page.
    """)

#About Page    
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
    ### About Dataset
    #### This dataset is recreated using offline augmentation from the original dataset.
    #### This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.
    #### A new directory containing 33 test images is created later for prediction purpose.
    #### Content
    1. train (70295 images)
    2. test (31 images)
    3. validation (17572 images)             
                """)
    
# Detector Page
elif(app_mode=="Disease Detector"):
    st.header("Disease Detector")
    test_image=st.file_uploader("Choose an Image to detect :")
    if(st.button("Show Image")):
        st.image(test_image,use_column_width=True)

    if(st.button("Predict")):
        with st.spinner("Please Wait.."):
            # st.write("Our Prediction")
            result_index=model_prediction(test_image)
            class_name=['Apple___Apple_scab','Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Blueberry___healthy',
            'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy',
            'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)',
            'Peach___Bacterial_spot',
            'Peach___healthy',
            'Pepper,_bell___Bacterial_spot',
            'Pepper,_bell___healthy',
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy',
            'Raspberry___healthy',
            'Soybean___healthy',
            'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch',
            'Strawberry___healthy',
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy']
            
            st.success(f"Our model predicted that it is {class_name[result_index]}")
            






