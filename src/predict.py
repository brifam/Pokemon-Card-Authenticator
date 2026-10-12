
import torch


from PIL import Image

from src.model import PokemonCardModelV1
from src.preprocessing import preprocess_image

# Select hardware
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Initialize and load trained model
model = PokemonCardModelV1(
    input_shape=3,
    hidden_units=32,
    output_shape=1
)

model.load_state_dict(
    torch.load(
        "models/pokemon_card_model_v1_updated1.pth",
        map_location=device,
        weights_only=True
    )
)

model.to(device)
model.eval()



def predict_card(image: Image.Image):
    """
    Accepts a PIL image and returns a prediction
    label and confidence percentage.
    """

    # Convert image to RGB
    tensor = preprocess_image(image).to(device)

    # Run inference
    with torch.inference_mode():
        outputs = model(tensor)
        raw_score = torch.sigmoid(outputs).item()

    # Determine prediction
    if raw_score >= 0.75:
        prediction = "Authentic"
        confidence = round(raw_score * 100, 2)

    elif raw_score <= 0.40:
        prediction = "Fake / Non-Pokémon Card"
        confidence = round((1 - raw_score) * 100, 2)

    else:
        prediction = "Inconclusive / High Uncertainty"
        confidence = round(raw_score * 100, 2)

    return prediction, confidence
