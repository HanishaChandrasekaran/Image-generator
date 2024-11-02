from flask import Flask, render_template, request, url_for, send_from_directory
from diffusers import StableDiffusionPipeline
import torch
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Create a Flask app
app = Flask(__name__)

# Folder to save generated images
output_folder = "generated_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    logging.info(f"Created folder: {output_folder}")

# Load the Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe.to("cpu")  # Use CPU, or "cuda" if you have a compatible GPU

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to serve generated images
@app.route('/generated_images/<filename>')
def get_generated_image(filename):
    return send_from_directory(output_folder, filename)

# Route to handle image generation
@app.route('/generate', methods=['POST'])
def generate_image():
    prompt = request.form['prompt']
    try:
        logging.info(f"Received prompt: {prompt}")

        # Generate the image
        image = pipe(prompt).images[0]
        logging.info("Image generated successfully.")

        # Save the image to the output folder
        image_filename = "generated_image.png"
        image_path = os.path.join(output_folder, image_filename)
        image.save(image_path)
        logging.info(f"Image saved at {image_path}")

        # Return the rendered template with the image path
        return render_template('index.html', image_path=url_for('get_generated_image', filename=image_filename))
    
    except Exception as e:
        logging.error(f"Error during image generation: {e}")
        return render_template('index.html', error="An error occurred during image generation.")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
