import numpy as np  # Importing a library for math stuff
from sklearn.cluster import KMeans  # Importing a library for clustering colors
import streamlit as st  # Importing a library for creating the web app
from PIL import Image  # Importing a library for image processing
from io import BytesIO  # Importing a library for handling byte streams (used for image saving)

# Function to compress the image using K-means clustering
def compress_image(image, k):
    data = np.array(image).reshape((-1, 3))  # Reshape image data to a 2D array
    kmeans = KMeans(n_clusters=k)  # Create KMeans object with k clusters
    kmeans.fit(data)  # Fit KMeans to the data
    compressed_data = kmeans.cluster_centers_[kmeans.labels_]  # Replace each pixel with its cluster center
    compressed_image = compressed_data.reshape(image.size[1], image.size[0], 3).astype(np.uint8)  # Reshape to original size
    return compressed_image, kmeans.cluster_centers_

# Function to resize the image
def resize_image(image, size):
    return image.resize(size, Image.LANCZOS)  # Resize image with high-quality filter

st.title("Pixel-Reducer")  # Title of the web app

# Allow user to upload an image file
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Slider to select number of colors
k = st.slider("Select number of Colors", min_value=1, max_value=32, value=8)

# If an image is uploaded
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)  # Open the image

        # Convert image to RGB if it has transparency
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Inputs for desired width and height
        width = st.number_input('Enter desired width', min_value=1, value=image.width)
        height = st.number_input('Enter desired height', min_value=1, value=image.height)
        
        resized_image = resize_image(image, (width, height))  # Resize the image
        
        # Save the original image to a buffer (in memory)
        original_buffer = BytesIO()
        resized_image.save(original_buffer, format="JPEG", quality=100)
        original_size = original_buffer.tell()  # Get the size of the original image
        
        # Compress the image
        compressed_image, compressed_colors = compress_image(resized_image, k)
        
        # Save the compressed image to a buffer
        compressed_image_pil = Image.fromarray(compressed_image)
        compressed_buffer = BytesIO()
        compressed_image_pil.save(compressed_buffer, format="JPEG", quality=70)
        compressed_size = compressed_buffer.tell()  # Get the size of the compressed image
        
        reduction_percentage = (original_size - compressed_size) / original_size * 100  # Calculate size reduction

        # Display images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(resized_image, caption='Original Image', use_column_width=True)
            st.write(f"Original size: {original_size / 1024:.2f} KB")
        with col2:
            st.image(compressed_image, caption='Compressed Image', use_column_width=True)
            st.write(f"Compressed size: {compressed_size / 1024:.2f} KB")
            st.write(f"Size reduced by: {reduction_percentage:.2f}%")

        compressed_buffer.seek(0)  # Go back to the beginning of the buffer
        st.download_button(
            label="Download Compressed Image",
            data=compressed_buffer,
            file_name="compressed_image.jpg",
            mime="image/jpeg"
        )
        st.success("Image compressed successfully!")  # Show success message
    
    except Exception as e:
        # Show error message if an error occurs
        if str(e) == "cannot write mode RGBA as JPEG":
            st.error("The uploaded image has transparency which is not supported by JPEG format. Please use an image without transparency.")
        else:
            st.error(f"An unexpected error occurred during compression: {str(e)}. Please try a different image or configuration.")

