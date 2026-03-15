import io
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

breedmodel = tf.keras.models.load_model('Breed_multiclass_100.h5')
agemodel = tf.keras.models.load_model('Age_multiclass_t14.h5')

def preprocess_image(image):
    img = image.resize((299, 299))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Define the API endpoint that accepts an image and returns the predicted dog breed
@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))
    preprocessed_image = preprocess_image(image)

    # Predict breed
    predictions = breedmodel.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions[0])
    
    max_prediction = predictions[0][predicted_class_index]
    if max_prediction < 0.2:
        return jsonify({'error': 'Invalid image'})

    class_names = np.array(['Chihuahua', 'Japanese Spaniel', 'Maltese','Pekinese', 'Shih Tzu', 'Blenheim Spaniel','Papillon', 'Toy terrier', 'Rhodesian ridgeback','Afghan hound', 'Basset','Beagle', 'Bloodhound', 'Bluetick','Black&Tan coonhound', 'Walker hound', 'English foxhound','Redbone', 'Italian greyhound', 'Whippet', 'Norwegian elkhound' , 'Otter hound' , 'Saluki' , 'Weimaraner' , 'Staffordshire bullterrier' , 'American Staffordshire terrier' , 'Border terrier' , 'Irish terrier' , 'Norfolk terrier' , 'Yorkshire terrier' , 'Wirehaired foxterrier' , 'Lakeland terrier' , 'Cairn' , 'Australian terrier' , 'Dandie Dinmont' , 'Boston bull' , 'Miniature schnauzer' , 'Standard schnauzer' , 'Scotch terrier' , 'Tibetan terrier' , 'Silky terrier' , 'Softcoated wheatenterrier' , 'WestHighland whiteterrier' , 'Lhasa' , 'Flatcoated retriever' , 'Curlycoated retriever' , 'Golden retriever' , 'Labrador retriever' , 'Chesapeake Bayretriever' , 'German shortHaired pointer' , 'Vizsla' , 'English setter' , 'Irish setter' , 'Gordon setter' , 'Brittany spaniel' , 'Clumber' , 'English springer wolfhound' , 'Welsh springer spaniel' , 'Cocker spaniel' , 'Sussex spaniel' , 'Irish waterspaniel' , 'Kuvasz' , 'Schipperke' , 'Malinois' , 'Briard' , 'Kelpie' , 'Komondor' , 'OldEnglish sheepdog' , 'Shetland sheepdog' , 'Collie' , 'Border collie' , 'Rottweiler' , 'German shepherd' , 'Doberman' , 'Miniature pinscher' , 'Greater SwissMountain' , 'Appenzeller' , 'EntleBucher' , 'Boxer' , 'Bull mastiff' , 'French bull' , 'Great Dane' , 'Saint Bernard' , 'Malamute' , 'Siberian husky' , 'Basenji' , 'Pug' , 'Leonberg' , 'Great Pyrenees' , 'Samoyedr' , 'Pomeranian' , 'Chow' , 'Keeshond' , 'Brabancon griffon' , 'Pembroke' , 'Cardigan' , 'Toy poodle' , 'Miniature poodle' , 'Mexican hairless' , 'Dingo'])
    predicted_breed = class_names[predicted_class_index]
    percentage = round(max_prediction * 100, 2)

    # Predict age
    age_predictions = agemodel.predict(preprocessed_image)
    age_predicted_class_index = np.argmax(age_predictions[0])
    age_class_names = np.array(['Adult','Puppy', 'Senior', 'Young'])
    predicted_age = age_class_names[age_predicted_class_index]

    # Return a JSON response
    response = {'breed': predicted_breed, 'breed_percentage': percentage, 'age': predicted_age}
    return jsonify(response)
     

if __name__ == '__main__':
    app.run(debug=True)