# Path: face_recognition_system/main.py

import numpy as np
import cv2
from mtcnn import MTCNN
from keras.models import load_model
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
import pickle

# Constants
MODEL_PATH = 'facenet_keras.h5'
DATASET_PATH = 'dataset/'  # Path to the dataset
EMBEDDINGS_PATH = 'embeddings.pickle'
CLASSIFIER_PATH = 'classifier.pickle'

# Load FaceNet model
def load_facenet_model(model_path):
    return load_model(model_path)

# Preprocess the input image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Detect faces using MTCNN
def detect_faces(image):
    detector = MTCNN()
    faces = detector.detect_faces(image)
    return faces

# Extract face embeddings using FaceNet model
def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    face_pixels = np.expand_dims(face_pixels, axis=0)
    embedding = model.predict(face_pixels)
    return embedding[0]

# Create embeddings for the dataset
def create_embeddings(model, dataset_path):
    embeddings = []
    labels = []
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_path):
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                image = preprocess_image(image_path)
                faces = detect_faces(image)
                if len(faces) > 0:
                    x, y, w, h = faces[0]['box']
                    face = image[y:y+h, x:x+w]
                    face = cv2.resize(face, (160, 160))
                    embedding = get_embedding(model, face)
                    embeddings.append(embedding)
                    labels.append(person_name)
    return np.array(embeddings), np.array(labels)

# Train classifier on embeddings
def train_classifier(embeddings, labels):
    in_encoder = Normalizer(norm='l2')
    embeddings = in_encoder.transform(embeddings)
    model = SVC(kernel='linear', probability=True)
    model.fit(embeddings, labels)
    return model

# Save embeddings and classifier
def save_embeddings_and_classifier(embeddings, labels, classifier):
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump((embeddings, labels), f)
    with open(CLASSIFIER_PATH, 'wb') as f:
        pickle.dump(classifier, f)

# Load embeddings and classifier
def load_embeddings_and_classifier():
    with open(EMBEDDINGS_PATH, 'rb') as f:
        embeddings, labels = pickle.load(f)
    with open(CLASSIFIER_PATH, 'rb') as f:
        classifier = pickle.load(f)
    return embeddings, labels, classifier

# Recognize faces in a new image
def recognize_faces(image_path, model, classifier):
    image = preprocess_image(image_path)
    faces = detect_faces(image)
    for face in faces:
        x, y, w, h = face['box']
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))
        embedding = get_embedding(model, face)
        embedding = embedding.reshape(1, -1)
        in_encoder = Normalizer(norm='l2')
        embedding = in_encoder.transform(embedding)
        prediction = classifier.predict(embedding)
        print(f'Predicted: {prediction[0]}')

# Main function
if _name_ == '_main_':
    model = load_facenet_model(MODEL_PATH)
    
    # Uncomment the following lines to create and train the model on a new dataset
    # embeddings, labels = create_embeddings(model, DATASET_PATH)
    # classifier = train_classifier(embeddings, labels)
    # save_embeddings_and_classifier(embeddings, labels, classifier)
    
    # Load existing embeddings and classifier
    embeddings, labels, classifier = load_embeddings_and_classifier()
    
    # Recognize faces in a new image
    test_image_path = 'test.jpg'  # Path to the test image
    recognize_faces(test_image_path, model, classifier)
