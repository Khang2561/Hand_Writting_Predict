import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

# Function to load the model
def load_model():
    l1 = torch.nn.Linear(784, 800)
    l2 = torch.nn.Linear(800, 10)
    model = torch.nn.Sequential(l1, torch.nn.ReLU(), l2)
    model.load_state_dict(torch.load("trained_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Function to preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image).unsqueeze(0)
    return image

# Function to make predictions
def predict_image(image, model):
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        return prediction.item(), probabilities.squeeze().tolist()

# Main function
def main():
    st.title("MNIST Digit Recognition")
    st.write("Upload a digit image and I will predict the number!")
    
    # Load the model
    model = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess the image and make prediction
        preprocessed_image = preprocess_image(image)
        prediction, probabilities = predict_image(preprocessed_image, model)
        
        st.write("")
        st.write(f"Prediction: {prediction}")
        st.write("Probabilities:")
        for i, prob in enumerate(probabilities):
            st.write(f"Number {i}: {prob:.4f}")
        

if __name__ == "__main__":
    main()
