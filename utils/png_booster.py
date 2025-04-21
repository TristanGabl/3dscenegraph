import cv2
import os

# Define the input and output directories
input_dir = '/teamspace/studios/this_studio/small_3DScannerApp_export/'
output_dir = '/teamspace/studios/this_studio/3dscenegraph/output/conf_boosted_images'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through the image filenames
for i in range(0, 762):
    # Construct the filename
    filename = f'conf_{i:05d}.png'
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    # Read the image
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Could not load image {input_path}.")
        continue

    # Boost the pixel values
    # Increase brightness by adding a constant value (e.g., 50)
    boosted_image = cv2.convertScaleAbs(image, alpha=100, beta=0)

    # Alternatively, increase contrast by multiplying by a constant factor (e.g., 1.5)
    # boosted_image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)

    # Save the boosted image
    cv2.imwrite(output_path, boosted_image)
    print(f"Boosted image saved to {output_path}")