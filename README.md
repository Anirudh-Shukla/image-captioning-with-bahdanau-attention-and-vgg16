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
- Check out the Kaggle: [Image Captioning with Bahdanau Attention and VGG16](https://www.kaggle.com/code/anirudhshukla1011/image-captioning-with-bahdanau-attention-and-vgg16/)
- Check out the deployed application: [Image Captioning with Bahdanau Attention and VGG16](https://automated-image-captioning.streamlit.app/)

## Contributors:
<a href="https://github.com/Anirudh-Shukla">ANIRUDH SHUKLA</a><br>
<img src="https://avatars.githubusercontent.com/u/136250552?v=4" width="50" height="50" alt="description"><br>
<a href="https://github.com/gauravkumarchaurasiya">GAURAV KUMAR CHAURASIYA</a><br>
<img src="https://avatars.githubusercontent.com/u/99001707?v=4" width="50" height="50" alt="description"><br>
<a href="https://github.com/SHIVANANAND">SHIVAN ANAND</a><br>
<img src="https://avatars.githubusercontent.com/u/137916628?v=4" width="50" height="50" alt="description"><be>

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

## Result
![image](https://github.com/user-attachments/assets/b13de4c0-717d-4971-83c9-7f637f99f2fc)
![image](https://github.com/user-attachments/assets/0a506392-4164-479e-89d1-3f9dbdb4cb01)

## Acknowledgements
- Various research papers and online tutorials on image captioning and attention mechanisms inspire this project. 
- Special thanks to the authors and the open-source community.
