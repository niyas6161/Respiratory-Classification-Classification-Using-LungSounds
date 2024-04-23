import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, GRU, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Orthogonal

classes = ["COPD", "Bronchiolitis", "Pneumonia", "URTI", "Healthy"]

# Custom GRU cell with Orthogonal initializer
def custom_gru(units):
    return GRU(units, recurrent_initializer=Orthogonal())

# Define the model architecture
def build_model(input_shape, num_classes):
    Input_Sample = Input(shape=input_shape)

    model_conv = Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu')(Input_Sample)
    model_conv = MaxPooling1D(pool_size=2, strides=2, padding='same')(model_conv)
    model_conv = BatchNormalization()(model_conv)

    model_gru_1 = GRU(128, return_sequences=True, activation='tanh', go_backwards=True)(model_conv)

    model_concat = concatenate([model_conv, model_gru_1])

    model_dense_1 = Dense(128, activation='relu')(model_concat)
    model_output = Dense(num_classes, activation='softmax')(model_dense_1)

    model = Model(inputs=Input_Sample, outputs=model_output)
    return model

# Function for loading and predicting with the model
def predict_class(model, classes, uploaded_file, features=52):
    if uploaded_file is not None:
        # Load audio file from file uploader
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        # Load audio data using librosa
        data_x, sampling_rate = librosa.load(uploaded_file.name, res_type='kaiser_fast')
        # Extract features
        mfccs = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=features).T, axis=0)
        # Reshape input data to match model's input shape
        val = np.expand_dims(np.expand_dims(mfccs, axis=0), axis=0)
        # Perform prediction
        prediction = classes[np.argmax(model.predict(val))]
        return prediction
    else:
        return "No file uploaded"

# Streamlit UI
def main():
    st.title("Respiratory Disease Diagnosis")
    st.sidebar.title("Options")

    # Build the model
    model = build_model((None, 52), len(classes))
    
    # Load weights
    model.load_weights("/home/niyas/respiratory/lung_sounds_new_org.h5")

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload audio file", type=["wav"])
    
    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav", start_time=0)

        # Perform prediction when file is uploaded
        if st.sidebar.button("Predict"):
            prediction = predict_class(model, classes, uploaded_file)
            st.write(f"Predicted class: {prediction}")

if __name__ == "__main__":
    main()
