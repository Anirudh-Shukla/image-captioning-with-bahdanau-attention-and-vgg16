# Visionary Narratives: Image Captioning with Bahdanau Attention and VGG16 Transfer Learning

## Overview
- This project focuses on generating image captions using a deep learning model that combines the VGG16 Convolutional Neural Network (CNN) for feature extraction and the Bahdanau Attention mechanism for improved caption generation.
- The attention mechanism allows the model to focus on different parts of an image while generating the caption, leading to more accurate and contextually relevant descriptions.

## Architecture
- VGG16: A pre-trained VGG16 model extracts high-level features from images. The last fully connected layers are removed, and the output from the final convolutional layer is used.
- Bahdanau Attention: This mechanism is applied to the features extracted from VGG16 to provide a weighted context vector that the LSTM uses to generate each word in the caption.
- LSTM: A Long Short-Term Memory network generates the sequence of words (caption) based on the context vector and previously generated words.

## Dataset
- The model is trained and evaluated on the Flickr8k dataset, which consists of 8,000 images each paired with five different captions. The dataset is divided into training, validation, and test sets.

## Dependencies
- TensorFlow
- Keras
- NumPy
- Streamlit
- PIL
- Pickle

## Installation
To run this project locally, follow these steps:

- Clone the repository:
```sh
git clone https://github.com/your-username/your-repository.git
cd your-repository
 ```

- Create a virtual environment:
```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
 ```

- Install the required packages:
```sh
pip install -r requirements.txt
 ```

- Run the Streamlit application:
```sh
streamlit run app.py
 ```
## Some Related Links
- Check out the Kaggle: [Image Captioning with Bahdanau Attention and VGG16}(https://www.kaggle.com/code/anirudhshukla1011/image-captioning-with-bahdanau-attention-and-vgg16/)
- Check out the deployed application: [Automated Image Captioning](https://automated-image-captioning.streamlit.app/)
