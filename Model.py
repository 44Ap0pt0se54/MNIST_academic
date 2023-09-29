import requests
import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

repository_owner = "44Ap0pt0se54"
repository_name = "MNIST_academic"
release_tag = "modelKeras"

# Construct the API URL for the release
api_url = f"https://api.github.com/repos/44Ap0pt0se54/MNIST_academic/releases/tags/modelKeras"
# Make a GET request to the API
response = requests.get(api_url)
# Check if the request was successful
if response.status_code == 200:
    release_info = response.json()
else:
    print(f"Failed to fetch release information. Status code: {response.status_code}")
    exit()
# Define a directory where you want to save the downloaded assets
download_directory = "/content"
# Create the directory if it doesn't exist
os.makedirs(download_directory, exist_ok=True)
# Iterate through the assets and download them
for asset in release_info['assets']:
    asset_name = asset['name']
    asset_url = asset['browser_download_url']
    download_path = os.path.join(download_directory, asset_name)
    print(f"Downloading {asset_name} from GitHub repo...")
    # Make a GET request to download the asset
    asset_response = requests.get(asset_url)
    if asset_response.status_code == 200:
        with open(download_path, 'wb') as f:
            f.write(asset_response.content)
        print(f"Downloaded {asset_name}")
    else:
        print(f"Failed to download {asset_name}. Status code: {asset_response.status_code}")

class Model:

    def __init__(self, type):

        self.type = type
        if type == "DNN":
            self.model = tf.keras.models.load_model("/content/DNN")
        elif type == "CNN":
            self.model = tf.keras.models.load_model("/content/CNN")
        elif type == "ResNet":
            self.model = tf.keras.models.load_model("/content/ResNet")
        else:
            print("Please verify syntaxe of input: type model")

    def test_model(self):

        # test dataset load
        test_set = np.load('test_set.npz')
        test_images, test_labels = test_set['x'], test_set['y']
        # test data normalization
        test_images = test_images / -255.0
        # evaluation on test data
        test_loss, test_acc = self.model.evaluate(test_images,  test_labels, verbose=2)
        print("Test accuracy = ", test_acc)

    def feature_visualization(self, layer_position):

        # test dataset load
        test_set = np.load('test_set.npz')
        test_images, test_labels = test_set['x'], test_set['y']

        if layer_position is not None:
        # Create a new model that includes layers up to the desired layer
            intermediate_model = tf.keras.Sequential(self.model.layers[:layer_position + 1])
        else:
            raise ValueError(f"You must select at least one layer")
        
        print("Output images of the first fiftheen kernels of layer "+str(layer_position)+" for the first five test images")

        for i in range(5):
            output_of_layer = intermediate_model(test_images)
            plt.figure(figsize=(15, 3)) 
            plt.subplots_adjust(wspace=0.1, hspace=0.0)  
    
            for j in range(15):
                plt.subplot(1, 16, j+2)  
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(output_of_layer[i][:, :, j], cmap='viridis')

            plt.subplot(1, 16, 1)  
            plt.axis('off')  
            plt.text(0, 0.5, str(test_labels[i]), fontsize=12, verticalalignment='center')
            plt.tight_layout()
        plt.show()


