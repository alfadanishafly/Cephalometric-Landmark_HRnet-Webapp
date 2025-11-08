from flask import Flask, request, render_template, redirect, url_for, flash
import os
import torch
import torch.nn as nn
import cv2
from torchvision import transforms
from timm import create_model
import numpy as np
import matplotlib
from PIL import Image
import base64

matplotlib.use('Agg')  # Menggunakan backend yang tidak memerlukan GUI
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import json

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'static/uploads'


def prediksi_dan_visualisasikan(image_path, model_path):
    # Muat model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hrnet = create_model('hrnet_w18', pretrained=False)
    hrnet.classifier = nn.Linear(hrnet.classifier.in_features, 32 * 2)

    # Load state dict without conv1 layer if it exists
    state_dict = torch.load(model_path, map_location=device)
    if 'conv1.weight' in state_dict:
        state_dict.pop('conv1.weight')
    if 'conv1.bias' in state_dict:
        state_dict.pop('conv1.bias')
    hrnet.load_state_dict(state_dict, strict=False)

    # Create new first conv layer to accept a single-channel grayscale image
    first_conv_layer = hrnet.conv1
    new_first_conv_layer = nn.Conv2d(1, first_conv_layer.out_channels,
                                     kernel_size=first_conv_layer.kernel_size,
                                     stride=first_conv_layer.stride,
                                     padding=first_conv_layer.padding,
                                     bias=first_conv_layer.bias is not None)

    # Replace the original first layer with the modified layer
    hrnet.conv1 = new_first_conv_layer

    hrnet.to(device)
    hrnet.eval()

    # Muat dan praproses gambar
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    px = Image.open(image_path)
    width, height = px.size
    path = image_path
    path = path.replace("\\", '/')

    if image is None:
        print(f"Error: Gambar tidak dapat dibaca dari path {image_path}")
        return None, None

    image = cv2.resize(image, (256, 256))
    original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)

    # Prediksi
    with torch.no_grad():
        outputs = hrnet(image)
        predicted_landmarks = outputs.view(-1, 32, 2).cpu().numpy()

    # Visualisasi hasil prediksi
    x_coords_pred = predicted_landmarks[0][:, 0]
    y_coords_pred = predicted_landmarks[0][:, 1]

    fig, ax = plt.subplots(figsize=(12, 16))
    ax.imshow(original_image)
    ax.scatter(x_coords_pred, y_coords_pred, c='r', label='Predicted')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Predicted Landmarks')
    ax.legend()

    # Simpan visualisasi ke dalam file sementara
    temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_visualization.png')
    plt.savefig(temp_image_path)
    plt.close(fig)

    return predicted_landmarks, temp_image_path.replace('static/', ''), temp_image_path, x_coords_pred.tolist(), y_coords_pred.tolist(), width, height, path


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "Tidak ada file yang diunggah"

        file = request.files['image']

        if file.filename == '':
            flash('Data Kosong', 'danger')

        if file:
            # Simpan file gambar ke server (misalnya, dalam folder "uploads")
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Baca data gambar
            predictions, image_path, temp_image_path, x_coords_pred, y_coords_pred, width, height, path = prediksi_dan_visualisasikan(filepath, 'model/cephalometric_model.pth')
            temp_image_path = temp_image_path.replace("\\", '/')
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if predictions is None:
                return "Error membaca gambar."
            else:
                return render_template('result coba.html', uploaded_image=temp_image_path,
                                       predictions=json.dumps(predictions.tolist()),
                                       temp_image_path=temp_image_path,
                                       x_coords_pred=x_coords_pred,
                                       y_coords_pred=y_coords_pred,
                                       original_image=filepath.replace('static/', ''),
                                       width=width, height=height,
                                       path=path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/dev')
def dev():
    return render_template('tes.html')


@app.route('/save_image', methods=['POST'])
def save_image():
    data = request.get_json()
    image_data = data['image']
    image_data = image_data.split(',')[1]
    image_data = base64.b64decode(image_data)
    filename = os.path.splitext(data['filename'])[0] + '_finalized.png'
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    with open(file_path, 'wb') as f:
        f.write(image_data)

    return {'message': 'Image saved successfully', 'path': file_path.replace('static/', '')}

if __name__ == '__main__':
    app.run(debug=True)
