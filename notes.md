# Experimentation with Model Log

## Custom CNN (PokemonCardModelV1)


Experiment 1: Using SGD with a higher learning rate

- Optimizer SGD
- LR: 0.1
- Epochs: 17
- Results: Val Accuracy was stuck at 71.21%

Notes: The results of the model hit a plateu only predicting the majority class. This is probably due to the SGD step size being too large causing it to overshoot or boounce.





Experiment 2: Using SGD with a lower learning rate but higher amount of epochs

- Optimizer: SGD
- LR : 0.001
- Epochs: 100
- Results: Val Accuracy stayed at around 57.86% and the training loss stayed at around 59%

Notes:
The bouncing stopped but the learning stopped also. Just plain SGD was overall too slow




Experiment 3: Change optimizer to Adam, changed the epoch amount to 15, and kept the learning rate

- Optimizer: Adam
- Learning rate .001
- Epochs: 15
- Results: Train Loss: 0.10, Train Accuracy: 97.50%
           Validation Loss: .08, Train Accuracy: 95.93%

Notes:
Changinng the optimizer quickly made it sucessful without bouncing. Saved weights to `models/pokemon_card_model_v1.pth`

### First Errors
- Although model trained with good accuracy, when testing it out on the web application, it detected counterfeit cards as real with high confidence

- Issue was most likely background of training dataset since on the website. The dataset had cards with gray backgrounds, so the website would prompt the user to crop the image to get rid of that issue. However the model was trained on
  the dataset with gray backgrounds... So issues with proportion and scales will most likely occur.

  ATTEMPT TO FIX: Cropped each image in the training and test datasets.

- Additionally, Flatten() to Linear Layer also might have issues in the classifier. The model was learning postion-specfic weights, so it associated certain feautes of a card to specific pixel locations rather than learning to detect those features  regardless of where they appeared. This problem is due to the fact the cropping is not exactly the same everytime the user crops and adds a card and the datasets have different croppings as well.

 ATTEMPT TO FIX: Adding AdaptiveAvgPool2d: takes the hidden_units * H * W and averages each channel down to a single number. Since pooling collapsed everything 1x1, only need one weight per channel.