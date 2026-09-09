from flask import Flask, render_template, request, jsonify
import torch 
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import io




class PokemonCardModelV1(nn.Module):
    def __init__(self,input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block1= nn.Sequential(
            nn.Conv2d(in_channels = input_shape, out_channels= hidden_units,kernel_size=3,padding=1, stride = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units, kernel_size=3, padding = 1, stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride = 2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_units, out_channels= hidden_units, kernel_size= 3, padding= 1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels= hidden_units, out_channels= hidden_units, kernel_size= 3, padding=1, stride =1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride =2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features= hidden_units*56*56, out_features= output_shape)
        )

    def forward(self,x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)


        return x



app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PokemonCardModelV1(input_shape=3, hidden_units=10, output_shape=1)
model.load_state_dict(torch.load('models/pokemon_card_model_v1.pth', map_location=device))
model.to(device)
model.eval()


card_transform = transforms.Compose([transforms.Resize((224,224), interpolation=InterpolationMode.BICUBIC),
                                    transforms.ToTensor()])


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400


    try:
        # Read image bytes directly without saving to disk
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Preprocess and pass through model
        tensor = card_transform(image).unsqueeze(0)
        tensor = tensor.to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.sigmoid(outputs)
            confidence = float(probabilities[0])
        
        # Placeholder response for testing connection:
        raw_score = probabilities[0].item()  # Sigmoid value between 0.0 and 1.0

        # Define threshold (Assuming 1 = Real, 0 = Fake)
        if raw_score >= 0.75:
            prediction = "Authentic"
            confidence = round(raw_score * 100, 2)
        elif raw_score <= 0.40:
            prediction = "Fake / Non-Pokémon Card "
            confidence = round((1 - raw_score) * 100, 2)
        else:
            prediction = "Inconclusive / High Uncertainty"
            confidence = round(raw_score * 100, 2)

        return render_template('index.html', result=prediction, confidence=confidence)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)