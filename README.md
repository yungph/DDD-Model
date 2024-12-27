Got it! Here's the updated description reflecting the repository name:

---

## DDD-model: Flask-Based Driver Drowsiness Detection System

This project is a Flask-based web application for detecting driver drowsiness using a custom-trained deep learning model. The application allows users to upload images, preprocess them, and receive predictions on whether the driver is drowsy or not.

### Features
- **Image Upload and Preprocessing:** Users can upload images, which are then preprocessed (converted to grayscale and resized) for model prediction.
- **Model Prediction:** Utilizes a custom-trained TensorFlow model to predict driver drowsiness.
- **Image Serving:** The application can serve the original and processed images.
- **Web Interface:** Includes HTML templates for user interaction.

### Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your_username/DDD-model.git
   cd DDD-model
   ```
2. **Run the Application:**
   ```bash
   python app.py
   ```

### Endpoints
- **`GET /`**: Renders the main HTML page.
- **`POST /upload`**: Uploads and preprocesses the image.
- **`POST /predict`**: Predicts the drowsiness status of the driver.
- **`GET /images/<filename>`**: Serves the images from the upload directory.
- **`GET /Start`**: Renders the Start page.

### File Structure
```
DDD-model/
├── app.py
├── Model
│   ├── ddd-detection-with-93-accur-final.ipynb
│   ├── DDD_model.h5
├── templates/
│   ├── index.html
│   ├── Start.html
├── static
|   ├──css
|      ├── index.css
|      ├── start.css
|   ├──js
|      ├── index.js
|      ├── start.js
|   ├── images
|       ├── moon.svg
|       ├── sun.svg
├── uploaded_images/


```
### DataSet
-https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset

### Preprocessing Functions
- **`preprocess_image(image_path, apply_histogram_equalization=False)`**: Preprocesses the image for display or further use.
- **`preprocess_image_for_model(image_path)`**: Preprocesses the image specifically for model prediction.

### Usage
1. Start the application:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to `http://localhost:5000`.
3. Upload an image and get the prediction results.

### Dependencies
- Flask
- TensorFlow
- Pillow (PIL)
- NumPy
- OpenCV
- Matplotlib
- Joblib

### Notes
- Ensure you have your trained model saved at the specified path.
- Adjust the model input dimensions in the preprocessing functions to match your model requirements.

### License
This project is licensed under the MIT License.
