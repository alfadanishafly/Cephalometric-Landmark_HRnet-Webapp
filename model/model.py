# model/model.py
import torch
from torchvision import transforms
from PIL import Image

def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

def predict(image_path, model):
    transform = transforms.Compose([  
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
    return outputs[0].numpy()  # Convert tensor to numpy array for easier handling
