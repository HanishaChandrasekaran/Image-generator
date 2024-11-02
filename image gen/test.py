from diffusers import StableDiffusionPipeline
import torch

# Load model and specify device
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe.to("cpu")  # Use CPU for testing (or "cuda" if you have a compatible GPU)

# Define prompt and output path
prompt = "A scenic view of trees"
output_path = "test_generated_image.png"  # Name of the file to save

try:
    # Generate and save the image
    print(f"Generating image for prompt: '{prompt}'...")
    image = pipe(prompt).images[0]
    image.save(output_path)
    print(f"Image saved successfully at {output_path}")
except Exception as e:
    print(f"Error during test generation: {e}")
