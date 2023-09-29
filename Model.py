import requests
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

github_DNN_model_url = ""
github_CNN_model_url = "https://github.com/44Ap0pt0se54/MNIST_academic/blob/main/%24CNNmodel.keras"
github_ResNet_model_url = ""

class Model:

    def __init__(self, type):

        self.type = type
        
        if type == "DNN":
            response = requests.get(github_DNN_model_url)
            if response.status_code == 200:
                with open("DNN_model.keras", "wb") as f:
                    f.write(response.content)
                    self.model = tf.keras.models.load_model("DNN_model.keras")
            else:
                print("Failed to download the DNN model.")

        elif type == "CNN":
            response = requests.get(github_DNN_model_url)
            if response.status_code == 200:
                with open("$CNN_model.keras", "wb") as f:
                    f.write(response.content)
                    self.model = tf.keras.models.load_model("$CNN_model.keras")
            else:
                print("Failed to download the CNN model.")

        elif type == "ResNet":
            response = requests.get(github_DNN_model_url)
            if response.status_code == 200:
                with open("ResNet_model.keras", "wb") as f:
                    f.write(response.content)
                    self.model = tf.keras.models.load_model("ResNet_model.keras")
            else:
                print("Failed to download the ResNet model.")
        else:
            print("Please verify syntaxe of input: type model")

    def test_model(self):

        # test dataset load 
        url = "https://github.com/44Ap0pt0se54/MNIST_academic/blob/main/test_set.npz"
        response = requests.get(url)
        with open("test_set.npz", "wb") as f:
            f.write(response.content)
        test_set = np.load('test_set.npz')
        test_images, test_labels = test_set['x'], test_set['y']

        # test data normalization
        test_images = test_images / -255.0

        # evaluation on test data
        test_loss, test_acc = self.model.evaluate(test_images,  test_labels, verbose=2)

        print("Test accuracy = ", test_acc)


    def feature_visualization(self, layer_position, test_images, test_labels):

        input_images = test_images

        if layer_position is not None:
        # Create a new model that includes layers up to the desired layer
            intermediate_model = tf.keras.Sequential(self.model.layers[:layer_position + 1])
        else:
            raise ValueError(f"You must select at least one layer")
        
        plt.figure(figsize=(10,10))
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            # Pass the input through the layers up to the desired layer
            for index in range(layer_position):
                output_of_layer = intermediate_model.layers[index](input_images)
                input_images = output_of_layer
            # Visualize the layer output (adjust as needed for your specific layer output)
            plt.imshow(output_of_layer[i], cmap='viridis')
            plt.xlabel(str(test_labels[i]))
            plt.show()




