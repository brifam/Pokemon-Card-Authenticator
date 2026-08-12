import torch
import torchvision
from PIL import Image
from model import PokemonCardModelV1
from torchvision import transforms


device = "cuda" if torch.cuda.is_available() else "cpu"
model = PokemonCardModelV1(input_shape=3, hidden_units=10, output_shape=1).to(device)

model.load_state_dict(torch.load("models/pokemon_card_model_v1.pth"))

model.eval()


card_transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])


def image_predictor(image_path):
    image = Image.open(image_path).convert("RGB")
    transformed_image = card_transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(transformed_image)
        pred_prob = torch.sigmoid(logits).item()
        if pred_prob > 0.5:
            pred_class = "Real"
            confidence = pred_prob
        else:
            pred_class= "Fake"
            confidence = 1.0 - pred_prob


        print(f"Prediction: {pred_class} | Confidence probability: {confidence*100:.2f}%")
        return pred_class, confidence




# Testing with demo images
uncropped_pred_class, uncropped_confidence =image_predictor("demo_images/uncropped_test.jpg")
cropped_pred_class, cropped_confidence =image_predictor("demo_images/cropped_test.jpg")


print(f"Uncropped Results: {uncropped_pred_class} | {uncropped_confidence*100:.2f}% Confidence")
print(f"Cropped Results: {cropped_pred_class} | {cropped_confidence*100:.2f}% Confidence")
