from flask import Flask, render_template, request, redirect
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.segmentation import find_boundaries
from sklearn.cluster import KMeans
import pickle

app = Flask(__name__)

pixel_size = 0.38  # mm/pixel

# Function to load and process grayscale MRI images
def load_grayscale_mri_image(image_path):
    image = plt.imread(image_path)
    grayscale_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    return grayscale_image

# Function to segment white matter using intensity thresholding
def segment_white_matter(grayscale_image, threshold_value=0.5):
    white_matter_mask = grayscale_image > threshold_value
    return white_matter_mask

# Function to calculate white matter density within an ROI
def calculate_white_matter_density(mask):
    white_matter_density = np.mean(mask)
    return white_matter_density

# Function to segment amygdala and calculate metrics
def segment_amygdala(grayscale_image, threshold_value=150):
    amygdala_binary_image = grayscale_image > threshold_value
    integer_image = amygdala_binary_image.astype(np.uint8)
    boundaries = find_boundaries(integer_image)
    region = regionprops(integer_image)[0]
    amygdala_area = region.area * 0.001
    amygdala_width = region.major_axis_length * 0.001
    return amygdala_area, amygdala_width

# Function to calculate amygdala metrics
def calculate_amygdala_metrics(amygdala_binary_image, pixel_size=0.38):
    amygdala_labeled_image = label(amygdala_binary_image)
    amygdala_areas = [region.area for region in regionprops(amygdala_labeled_image)]
    amygdala_area_pixels = sum(amygdala_areas)
    amygdala_area_mm2 = amygdala_area_pixels * pixel_size**2
    amygdala_area_cm3 = amygdala_area_mm2 / 100
    amygdala_major_axis_length_pixels = max(region.major_axis_length for region in regionprops(amygdala_labeled_image))
    amygdala_width_mm = amygdala_major_axis_length_pixels * pixel_size
    return amygdala_area_mm2, amygdala_area_cm3, amygdala_width_mm

# Function to calculate hippocampus volume
def calculate_hippocampus_volume(image_path):
    image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    hippocampus_area = cv2.contourArea(largest_contour)
    hippocampus_volume = hippocampus_area / 1000  # Convert to mm^3
    return hippocampus_volume

# Function to segment gray matter using KMeans clustering
def segment_gray_matter(grayscale_image):
    kmeans = KMeans(n_clusters=2, n_init=10)  # Explicitly set n_init
    labels = kmeans.fit_predict(grayscale_image.reshape(-1, 1))
    gray_matter_density = np.mean(labels == 1)
    return gray_matter_density

# Function to segment cortex using KMeans clustering
def segment_cortex(grayscale_image):
    kmeans = KMeans(n_clusters=2, n_init=10)  # Explicitly set n_init
    labels = kmeans.fit_predict(grayscale_image.reshape(-1, 1))
    cortex_pixel_count = np.sum(labels == 1)
    voxel_volume_mm3 = 0.43 * 0.43 * 0.43  # Adjust pixel size to actual dimensions
    cortex_volume_mm3 = cortex_pixel_count * voxel_volume_mm3
    return cortex_volume_mm3

# Function to segment thalamus and calculate volume
def segment_thalamus(grayscale_image):
    threshold = np.percentile(grayscale_image, 95)
    binary_image = grayscale_image > threshold
    labeled_image = label(binary_image)
    regions = regionprops(labeled_image)
    largest_region = max(regions, key=lambda region: region.area)
    voxel_size_cm3 = 0.007  # Adjust voxel size to actual dimensions
    thalamic_volume_cm3 = largest_region.area * voxel_size_cm3
    return thalamic_volume_cm3

# Function to extract features from an image
def extract_features(image_path):
    grayscale_mri_image = load_grayscale_mri_image(image_path)

    # Segment and calculate amygdala metrics
    amygdala_area, amygdala_width = segment_amygdala(grayscale_mri_image)
    amygdala_area_mm2, amygdala_area_cm3, amygdala_thickness = calculate_amygdala_metrics(grayscale_mri_image)

    # Segment and calculate white matter density
    white_matter_mask = segment_white_matter(grayscale_mri_image)
    white_matter_density = calculate_white_matter_density(white_matter_mask)

    # Segment and calculate gray matter density
    gray_matter_density = segment_gray_matter(grayscale_mri_image)

    # Segment and calculate cortex volume
    cortex_volume = segment_cortex(grayscale_mri_image)

    # Segment and calculate thalamic volume
    thalamic_volume = segment_thalamus(grayscale_mri_image)

    # Calculate and return hippocampus volume
    hippocampus_volume = calculate_hippocampus_volume(image_path)

    return amygdala_area_mm2, amygdala_area_cm3, amygdala_thickness, white_matter_density, gray_matter_density, cortex_volume, thalamic_volume, hippocampus_volume

# Example usage
@app.route("/", methods=['POST', 'GET'])
def home():
    return render_template('login.html')

@app.route("/login", methods=['POST', 'GET'])
def index():
    return render_template('index.html')

@app.route("/next_page", methods=['POST', 'GET'])
def next_page():
    return render_template('next_page.html')

@app.route('/upload', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file:
            # Ensure 'static/uploads/' directory exists
            uploads_dir = 'static/uploads/'
            if not os.path.exists(uploads_dir):
                os.makedirs(uploads_dir)

            # Save the uploaded file
            mri_image_path = os.path.join(uploads_dir, file.filename)
            file.save(mri_image_path)

            # Get the list of files in the 'static/uploads' directory
            uploaded_files = os.listdir(uploads_dir)

            # Get the last uploaded image
            last_uploaded_image = uploaded_files[-1]

            # Load the last grayscale image
            grayscale_image = load_grayscale_mri_image(os.path.join(uploads_dir, last_uploaded_image))

            # Extract features from the uploaded image
            features = extract_features(mri_image_path)

            # Create a dictionary to store the features
            feature_dict = {
                'Amygdala Area (mm^2)': features[0],
                'Amygdala Volume (cm^3)': features[1],
                'Amygdala Thickness (mm)': features[2],
                'White Matter Density': features[3],
                'Gray Matter Density': features[4],
                'Cortex Volume (mm^3)': features[5],
                'Thalamic Volume (cm^3)': features[6],
                'Hippocampus Volume (mm^3)': features[7]
            }
            normal_values = {
                'Amygdala Area (mm^2)': 4778.280135,
                'Amygdala Volume (cm^3)': 47.78280135,
                'Amygdala Thickness (mm)': 13.38620734,
                'White Matter Density': 0.791264518,
                'Gray Matter Density': 0.485479206,
                'Cortex Volume (mm^3)': 1994.326977,
                'Thalamic Volume (cm^3)': 4.877685714,
                'Hippocampus Volume (mm^3)': 5.520964796
            }

            # Subtract normal values from the calculated features
            # Subtract normal values from the calculated features and ensure positivity
            subtracted_features = [abs(feature - normal_values[key]) for key, feature in zip(feature_dict.keys(), features)]


            # Display the subtracted features
            result_list = {
                'Amygdala Area ': subtracted_features[0],
                'Amygdala Volume ': subtracted_features[1],
                'Amygdala Thickness ': subtracted_features[2],
                'White Matter Density': subtracted_features[3],
                'Gray Matter Density': subtracted_features[4],
                'Cortex Volume ': subtracted_features[5],
                'Thalamic Volume ': subtracted_features[6],
                'Hippocampus Volume ': subtracted_features[7]
            }

            # Prepare input for the model
            inp = np.array(list(feature_dict.values())).reshape(1, -1)

            # Load the trained model
            loaded_model = pickle.load(open('models/austin_model.sav', 'rb'))

            # Make predictions
            pred = loaded_model.predict(inp)

            # Determine the result
            result = 'normal' if pred[0] == 1 else 'autism'

            return render_template('result12.html', result=result, image_path=mri_image_path, features=result_list)

    return render_template('upload_form.html')

if __name__ == "__main__":
    app.run()
