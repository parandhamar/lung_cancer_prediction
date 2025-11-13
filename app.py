from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# ================= Flask App =================
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ================= Device =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================= Load Model =================
class_names = ['Bengin cases', 'Malignant cases', 'Normal cases']
model_path = "densenet121_final.pth"  # Place your trained model here

densenet121 = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
num_ftrs = densenet121.classifier.in_features
densenet121.classifier = nn.Linear(num_ftrs, len(class_names))
densenet121.load_state_dict(torch.load(model_path, map_location=device))
densenet121.to(device)
densenet121.eval()
print("✅ Model loaded successfully!")

# ================= Image Transform =================
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================= Prediction Function =================
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = data_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = densenet121(image)
        _, pred = torch.max(outputs, 1)
        return class_names[pred.item()]

# ================= Routes =================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        prediction = predict_image(file_path)
        return render_template('result.html', filename=file.filename, prediction=prediction)

# ✅ NEW: Route to display uploaded images in result.html
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ================= Run App =================
if __name__ == '__main__':
    app.run(debug=True)

