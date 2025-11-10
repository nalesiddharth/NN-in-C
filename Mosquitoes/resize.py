import os
from PIL import Image

# Input and output directories
input_dir = r"D:\DevEnv\NN-in-C\Mosquitoes\Original"
output_dir = r"D:\DevEnv\NN-in-C\Mosquitoes\Upscaled"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Target size
target_size = (512, 385)  # width x height

# Process all images in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".png")

        try:
            # Open the image
            img = Image.open(input_path)

            # Resize to 56x42
            img = img.resize(target_size)

            # Convert to 8-bit grayscale
            img = img.convert("L")

            # Save as PNG
            img.save(output_path, format="PNG")

            print(f"Processed: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("\nAll images processed successfully!")
