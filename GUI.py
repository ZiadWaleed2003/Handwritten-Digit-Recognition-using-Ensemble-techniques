import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from Logic import make_prediction

# 3. Streamlit GUI Functions
def render_header():
    """Render the title and instructions for the app."""
    st.title("MNIST Digit Predictor")
    st.markdown("Upload a 28x28 grayscale image of a digit (0-9) to predict its value. The image should ideally have a white digit on a black background, like the MNIST dataset.")

def render_model_selection():
    """Render the model selection dropdown and return the selected model choice."""
    model_choice = st.selectbox(
        "Select Model to Use:",
        options=[
            ("Stacking Model", 3),
            ("Bagging Model", 1),
            ("Boosting Model", 2)
        ],
        format_func=lambda x: x[0]
    )
    return model_choice[1]

def render_file_uploader():
    """Render the file uploader and color inversion checkbox, return the uploaded file and inversion choice."""
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    invert_colors = st.checkbox("Invert colors (if digit is black on white background)", value=False)
    return uploaded_file, invert_colors


def display_prediction(predicted_digit):
    """Display the predicted digit."""
    st.subheader("Prediction")
    st.markdown(f"**Predicted Digit:** {predicted_digit}")

def main():
    """Main function to orchestrate the Streamlit app."""
    # Sidebar for model selection and instructions
    with st.sidebar:
        st.header("Controls")
        render_header()
        model_choice = render_model_selection()

    # Main content area
    st.header("Image Upload and Prediction")
    col1, col2 = st.columns([1, 1])  # Two columns for layout

    with col1:
        # File uploader and inversion option
        uploaded_file, invert_colors = render_file_uploader()

    if uploaded_file is not None:
        with col1:
            # Display uploaded image
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)

        # Save the uploaded image temporarily
        temp_image_path = "temp_image.png"
        img.save(temp_image_path)

        # Process and predict
        with st.spinner("Processing image and making prediction..."):
            if invert_colors:
                img = img.convert('L')
                img_array = np.array(img)
                img_array = 255 - img_array  # Invert colors
                img = Image.fromarray(img_array.astype(np.uint8))
                img.save(temp_image_path)

            # Make the prediction
            predicted_digit = make_prediction(model_choice, temp_image_path)

        with col2:
            # Display preprocessed image and prediction
            img_array = np.array(Image.open(temp_image_path).convert('L'))

            display_prediction(predicted_digit)
    else:
        with col1:
            st.write("Please upload an image to get a prediction.")

# 4. Run the App
if __name__ == "__main__":
    main()