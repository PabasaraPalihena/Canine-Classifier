# 🐾 Canine-Classifier : Dog Breed & Age Identification models

A Flask-based REST API that uses custom-trained deep learning models to identify a dog's **breed** (including mixed breeds) and estimate its **age group** from a given image URL.

Built as part of an undergraduate research project exploring convolutional neural network (CNN) architectures for fine-grained visual classification of dogs.

## 📚 Research

- 📄 **Research Paper:** https://ieeexplore.ieee.org/document/10155781

- 🎥 **Final Product Demo Video:** https://drive.google.com/file/d/1Q0aLzC-xwHFS0sMtwmxcQzE3CHywx6H8/view

## ✨ Features

- **Breed Identification** — Classifies over **100 dog breeds**, including some common **mixed breed** combinations
- **Age Estimation** — Classifies a dog into one of four age groups: `Puppy`, `Young`, `Adult`, `Senior`
- **URL-based inference** — Accepts a publicly accessible image URL for prediction
- **Confidence scoring** — Returns breed prediction confidence as a percentage
- **Low-confidence rejection** — Returns `"Invalid image"` when the model confidence is below 20%

## 🏗️ Project Structure

```
flaskApp/
│
├── app.py                      # Main Flask API (breed + age prediction)
├── pp2_flask-api.py            # Alternate/prototype API version
├── test-multilabel.py          # Test script for multilabel breed model
├── test-mixedMulticlass.py     # Test script for mixed-breed multiclass model
└── README.md
```

## 🛠️ Tech Stack

- **Python 3.x**
- **Flask** — REST API framework
- **TensorFlow / Keras** — Model loading and inference
- **Pillow** — Image preprocessing
- **NumPy** — Array manipulation

## 📝 License

This project was developed for academic research purposes.
