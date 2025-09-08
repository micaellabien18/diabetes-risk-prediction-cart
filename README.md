# Diabetes Risk Prediction Web App (CART Algorithm)

A web application for predicting diabetes risk using the CART (Classification and Regression Trees) algorithm. Users can explore data visualizations, input health metrics for prediction, and contact the team for support.

> **Note:** This project demonstrates my foundational web development and machine learning skills from my Computer Science studies. It is presented in its original form as submitted for coursework. As I continue learning, I plan to refactor and enhance this project to reflect my growth and improved abilities.

## Features
- **Diabetes Risk Prediction:** Enter health data to receive instant risk assessment using a trained CART model.
- **Data Visualization:** Interactive plots for feature distributions and class balance.
- **Contact Form:** Send messages directly from the app (email integration).

## Usage
- **Home:** Learn about diabetes and the app's purpose.
- **Prediction:** Enter health metrics to get a risk assessment.
- **Data:** View model accuracy and feature visualizations.
- **Contact:** Send questions or feedback via the contact form.

## Technical Implementation

### Machine Learning
- **Algorithm:** CART Decision Tree Classifier
- **Dataset:** 750+ patient records with 8 health features
- **Libraries:** Scikit-learn, Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn for feature analysis

### Web Implementation
- **Backend:** Flask with RESTful API endpoints
- **Frontend:** HTML, CSS, JavaScript
- **Email:** Flask-Mail integration for contact forms
- **Data:** CSV-based dataset processing