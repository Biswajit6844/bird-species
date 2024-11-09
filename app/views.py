# Important imports
from app import app
from flask import request, render_template, url_for
from keras import models
import numpy as np
from PIL import Image
import string
import random
import os

# Adding path to config
app.config['INITIAL_FILE_UPLOADS'] = 'app/static/uploads'

# Loading model
model = models.load_model('app/static/model/bird_species.h5')

# Route to home page
@app.route("/", methods=["GET", "POST"])
def index():

    # Execute if request is GET
    if request.method == "GET":
        full_filename = 'images/white_bg.jpg'
        return render_template("index.html", full_filename=full_filename)

    # Execute if request is POST
    if request.method == "POST":
        try:
            # Check if file is uploaded
            if 'image_upload' not in request.files:
                error_message = "No file part in the request"
                return render_template("index.html", error=error_message)

            image_upload = request.files['image_upload']

            # Check if a file is selected
            if image_upload.filename == '':
                error_message = "No selected file"
                return render_template("index.html", error=error_message)

            # Generating unique image name
            letters = string.ascii_lowercase
            name = ''.join(random.choice(letters) for i in range(10)) + '.png'
            full_filename = 'uploads/' + name

            # Reading, resizing, saving, and preprocessing image for prediction
            image = Image.open(image_upload)
            image = image.resize((224, 224))
            image.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], name))
            image_arr = np.array(image.convert('RGB'))
            image_arr.shape = (1, 224, 224, 3)

            # Predicting output
            result = model.predict(image_arr)
            ind = np.argmax(result)
            classes = [
                'AMERICAN GOLDFINCH', 'BARN OWL', 'CARMINE BEE-EATER',
                'DOWNY WOODPECKER', 'EMPEROR PENGUIN', 'FLAMINGO'
            ]

            # Returning template with prediction result
            return render_template('index.html', full_filename=full_filename, pred=classes[ind])

        except Exception as e:
            # Log the error and return a default response
            error_message = f"An error occurred: {str(e)}"
            return render_template("index.html", error=error_message)

# Main function
if __name__ == '__main__':
    app.run(debug=True)

