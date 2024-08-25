# Plant Disease Detection Using Deep Learning 🌱

This project involves the detection of plant diseases using deep learning techniques. The goal is to build a model that can identify different types of plant diseases from images of leaves. The project utilizes convolutional neural networks (CNNs) for image classification, implemented using TensorFlow and Keras.

## Features ✨

- **Deep Learning Model**: A convolutional neural network (CNN) trained to detect plant diseases from leaf images. 🧠
- **Data Augmentation**: Image augmentation techniques applied to improve model generalization. 🔄
- **Visualization**: Graphical representation of training and validation accuracy/loss. 📉
- **Streamlit Integration**: A user-friendly web interface for uploading images and getting predictions. 🌐

## Hosted Version 🌐

A hosted version of the Plant Shield is available at: [Plant Shield](https://plant-shield.streamlit.app/)

## Requirements 📦

- TensorFlow==2.12.0
- numpy==1.24.0
- matplotlib==3.7.0
- scikit-learn==1.2.2
- opencv-python==4.7.0
- seaborn==0.12.2
- streamlit==1.20.0

## Installation ⚙️

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/plant-disease-detection.git
    cd plant-disease-detection
    ```

2. **Install all requirements**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage 🚀

1. **Run the Streamlit app**:
    ```bash
    streamlit run gui.py
    ```

2. **Upload an image**: Use the interface to upload an image of a plant leaf from the test folder. 📸
3. **Get predictions**: The model will predict and display the type of disease (if any) present in the leaf. 🏷️

## Data 🗂️

The dataset used for this project consists of images of healthy and diseased plant leaves. The images are pre-processed and augmented to create a robust model. You can access the dataset using this [link](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).

## Model Training 🏋️

The model is trained using TensorFlow and Keras. The training script includes steps for loading data, augmenting images, building the CNN, and training the model.

## Results 📈

The trained model achieves high accuracy in detecting plant diseases and can be further improved with more data and fine-tuning.

## Contributions 🤝

Contributions to improve the model and add new features are welcome. Feel free to fork the repository and create a pull request.

## License 📝

This project is licensed under the MIT License.

## Sample Images 📸
![image](https://github.com/Akshat-Raii/Plant_Disease_Detection/assets/141046886/dcb1e8e1-6905-44be-bcf0-517fbf676abe)
![image](https://github.com/Akshat-Raii/Plant_Disease_Detection/assets/141046886/6f8e5e4c-3c63-4041-b42f-685d093db278)
![image](https://github.com/Akshat-Raii/Plant_Disease_Detection/assets/141046886/e508599b-a8ed-4788-ab2b-d9f3432cc186)
![image](https://github.com/Akshat-Raii/Plant_Disease_Detection/assets/141046886/bfd513d5-3323-48c9-836a-94f542e97bf4)
![image](https://github.com/Akshat-Raii/Plant_Disease_Detection/assets/141046886/5782040d-e32a-4446-be89-7a2ccaedcad9)
