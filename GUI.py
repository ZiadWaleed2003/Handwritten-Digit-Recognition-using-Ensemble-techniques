import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from Logic import make_prediction

# 3. Streamlit GUI Functions
def render_header():
    """Render the title and instructions for the app."""


    st.title("MNIST Digit Predictor")
    st.write("Upload a 28x28 grayscale image of a digit (0-9) to predict its value.")
    st.write("The image should ideally have a white digit on a black background, like the MNIST dataset.")

def render_model_selection():
    """Render the model selection dropdown and return the selected model choice."""


    model_choice = st.selectbox(
        "Select Model to Use:",
        options=[
            ("Stacking Model", 3),
            ("Bagging Model", 1),
            ("Boosting Model", 2)
        ],
        format_func=lambda x: x[0]  # Display only the model name in the dropdown
    )
    return model_choice[1]  # Return the model choice value (0, 1, or 2)

def render_file_uploader():
    """Render the file uploader and color inversion checkbox, return the uploaded file and inversion choice."""


    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    invert_colors = st.checkbox("Invert colors (if digit is black on white background)", value=False)
    return uploaded_file, invert_colors

def display_preprocessed_image(img_array):
    """Display the preprocessed image and preprocessing details."""


    st.subheader("Preprocessed Image (28x28 Grayscale)")
    fig, ax = plt.subplots()
    ax.imshow(img_array, cmap='gray')
    ax.axis('off')
    st.pyplot(fig)

    st.write("**Preprocessing Details:**")
    st.write(f"- Image shape after flattening: {(img_array.flatten().shape)}")
    st.write(f"- Pixel value range after normalization: {(img_array.min()/255.0):.2f} to {(img_array.max()/255.0):.2f}")

def display_prediction(predicted_digit):
    """Display the predicted digit."""


    st.subheader("Prediction")
    st.write(f"**Predicted Digit:** {predicted_digit}")

def main():
    """Main function to orchestrate the Streamlit app."""


    # Render the header
    render_header()

    # Render model selection
    model_choice = render_model_selection()

    # Render file uploader and color inversion option
    uploaded_file, invert_colors = render_file_uploader()

    if uploaded_file is not None:
        # Load and display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Save the uploaded image temporarily to pass its path to make_prediction
        temp_image_path = "temp_image.png"
        img.save(temp_image_path)

        # Process and predict
        with st.spinner("Processing image and making prediction..."):
            # Adjust the image if color inversion is needed
            if invert_colors:
                img = img.convert('L')
                img_array = np.array(img)
                img_array = 255 - img_array  # Invert colors
                img = Image.fromarray(img_array.astype(np.uint8))
                img.save(temp_image_path)  # Save the inverted image

            # Make the prediction
            predicted_digit = make_prediction(model_choice, temp_image_path)

        # # Display the preprocessed image and details
        # display_preprocessed_image(img_array)

        # Display the prediction
        display_prediction(predicted_digit)
    else:
        st.write("Please upload an image to get a prediction.")

# 4. Run the App
if __name__ == "__main__":
    main()