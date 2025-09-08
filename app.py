import pandas as pd
from flask_mail import Mail, Message
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__, static_url_path='/static', static_folder='static')

load_dotenv()

# Configuration for Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')

# Initialize Flask-Mail
mail = Mail(app)

plot_dir = 'static/plots'
os.makedirs(plot_dir, exist_ok=True)

# Load dataset
df = pd.read_csv('diabetes.csv')

# Split dataset into features and target variable
X = df.drop('Outcome', axis=1)
y = df['Outcome']

feature_names = X.columns

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# Instantiate the Decision Tree classifier
model = DecisionTreeClassifier()

# Train model on training data
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Function to send email
def send_email(name, subject, message):
    sender_email = os.getenv('MAIL_USERNAME')
    recipient_email = os.getenv('MAIL_RECIPIENT')

    msg = Message(subject,
                  sender=sender_email,
                  recipients=[recipient_email])
    msg.body = f"Name: {name}\nSubject: {subject}\nMessage: {message}"
    
    mail.send(msg)

def create_visualizations():
    color = (0/255, 136/255, 169/255)

    # Distribution of features
    for feature in feature_names:
        plt.figure(figsize=(8, 6))
        if feature == 'BloodPressure':
            sns.histplot(df[feature], kde=True, color=color, bins=20) 
            plt.title('Blood Pressure Distribution', fontsize=16) 
        elif feature == 'SkinThickness':
            sns.histplot(df[feature], kde=True, color= color)
            plt.title('Skin Thickness Distribution', fontsize=16) 
        elif feature == 'DiabetesPedigreeFunction':
            sns.histplot(df[feature], kde=True, color= color)
            plt.title('Diabetes Pedigree Function Distribution', fontsize=16)

        else:
            sns.histplot(df[feature], kde=True, color=color)
            plt.title(f'{feature} Distribution', fontsize=16)  # Title with spaces between words
        plt.xlabel(feature, fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/{feature}_distribution.png', bbox_inches='tight')
        plt.close()

    # Class balance
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Outcome', hue='Outcome', palette='Set1', legend=False)
    plt.title('Class Balance', fontsize=16)
    plt.xlabel('Outcome', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/class_balance.png', bbox_inches='tight')
    plt.close()

create_visualizations()

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        name = request.form['name']
        subject = request.form['subject']
        message = request.form['message']

        send_email(name, subject, message)
        
        return jsonify({'message': 'Email sent successfully!'})

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/data')
def data():
    train_accuracy = metrics.accuracy_score(y_train, model.predict(X_train))
    test_accuracy = metrics.accuracy_score(y_test, y_pred)
    return render_template('data.html', train_accuracy=train_accuracy, test_accuracy=test_accuracy)

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    pregnancies = data['pregnancies']
    glucose = data['glucose']
    bloodpressure = data['bloodpressure']
    skinthickness = data['skinthickness']
    insulin = data['insulin']
    bmi = data['bmi']
    diabetespedigreefunction = data['diabetespedigreefunction']
    age = data['age']

    # Prediction using trained model
    prediction = model.predict([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age]])[0] == 1

    result = { 'prediction': int(prediction) }
    return jsonify(result)

if __name__ == '__main__':
    app.run()