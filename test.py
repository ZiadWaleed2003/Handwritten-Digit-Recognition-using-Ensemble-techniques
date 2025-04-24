import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.title("Test Drawable Canvas")
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=20,
    stroke_color="black",
    background_color="#eee",
    height=150,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
)
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)