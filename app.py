from flask import Flask, render_template, request, jsonify
import os
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import pandas as pd
import numpy as np
import csv
from test import TableQuestionAnswering
import time

# Get absolute path to the folder where this file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Reading the dataset
disease_info = pd.read_csv(os.path.join(BASE_DIR, 'Model_assest', 'disease_info.csv'))
suppliment_info = pd.read_csv(
    os.path.join(BASE_DIR, 'Model_assest', 'supplement_info.csv'), 
    encoding='cp1252'
)

# Device agnostic code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create instance of the pretrained model
model = models.resnet18(pretrained=True)

# Modifying the layers
num_classes = 38
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load the model checkpoint
model_checkpoint_path = os.path.join(BASE_DIR, 'Model_assest', 'model.pth')
model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
model.eval()

# Define the transformation for input images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function for image prediction
def prediction(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    output = model(image)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

app = Flask(__name__)

# Define allowed networks
allowed_networks = ['192.168.83.', '192.168.160.', '192.168.146.', '192.168.0']

def is_allowed_network(ip):
    return any(ip.startswith(network) for network in allowed_networks)

# Table QA instance
tqa_instance = TableQuestionAnswering()
tqa_instance.load_table(os.path.join(BASE_DIR, 'Model_assest', 'DiseaseChatbotData.csv'))

@app.route('/', methods=['GET', 'POST'])
def home():
    answer = None
    if request.method == 'POST':
        query = request.form.get('user_input')
        answer = tqa_instance.answer_query(query)
    return render_template('home.html', answer=answer)

@app.route('/index')
def ai_detect_page():
    return render_template('index.html')

@app.route('/supplement')
def supplement():
    supplement_data = []
    with open(os.path.join(BASE_DIR, 'Model_assest', 'supplement_info.csv'), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            supplement_data.append({
                'supplement': row['supplement name'],
                'supplement_img': row['supplement image'],
                'supplement_prod_link': row['buy link']
            })
    return render_template('supplement.html', supplement_data=supplement_data)

@app.route('/submit', methods=['POST'])
def submit():
    if 'image' in request.files:
        image = request.files['image']
        filename = image.filename
        upload_folder = os.path.join(BASE_DIR, 'static', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)  # Ensure upload folder exists
        file_path = os.path.join(upload_folder, filename)
        image.save(file_path)

        # Perform prediction
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        supplement = suppliment_info['supplement name'][pred]
        supplement_img = suppliment_info['supplement image'][pred]
        supplement_prod_link = suppliment_info['buy link'][pred]

        image_url = f'/static/uploads/{filename}'

        response = {
            'prediction': title,
            'image': image_url,
            'discrption': disease_info['description'][pred],
            'possible_step': disease_info['Possible Steps'][pred],
            'supplement': supplement,
            'supplement_img': supplement_img,
            'supplement_name': suppliment_info['supplement name'][pred],
            'supplement_prod_link': supplement_prod_link
        }
        return render_template('submit.html', data=response)
    
    return jsonify({'error': 'Invalid request'})

@app.route('/response', methods=['GET', 'POST'])
def response():
    answer = ""
    query = ""
    if request.method == 'POST':
        query = request.form.get('text')
        time.sleep(2)
        answer = tqa_instance.answer_query(query)
    return render_template('chatbot.html', resp={"query": query, "answer": answer})

@app.route('/learnmore')
def learnmore():
    return render_template('learnmore.html')

if __name__ == '__main__':
    app.run(debug=True)
