import os 
import shutil
import pandas as pd

def image_sort(labels, source_directory,target_directory):

    """
    Sorts the card images into their labels and their designated folders.
    
    """

    df = pd.read_csv(labels)


    os.makedirs(os.path.join(target_directory,"real"), exist_ok= True)
    os.makedirs(os.path.join(target_directory, "fake"), exist_ok = True)

    for _, row in df.iterrows():
        img_id = str(row["id"])
        if row["label"] == 1:
            label = "real"
        else:
            label = "fake"

        filename = f"{img_id}.JPG"
        source = os.path.join(source_directory, filename)
        destination = os.path.join(target_directory, label, filename)


        if os.path.exists(source):
            shutil.copy(source,destination)


# Running the Functions

image_sort("downloaded_dataset/train_labels.csv","downloaded_dataset/train", "dataset/train")

image_sort("downloaded_dataset/test_labels.csv","downloaded_dataset/test", "dataset/test")

print("Dataset sorting completed.")