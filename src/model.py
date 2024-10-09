from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout

def create_model(
    input_shape, 
    n_categories, 
    activation = "softmax"
):
    """
    Create a Convolutional Neural Network (CNN) model for audio classification.

    This function builds a sequential CNN model with multiple convolutional and pooling layers,
    followed by dense layers for classification.

    Parameters:
    input_shape (tuple): The shape of the input data (height, width, channels).
    n_categories (int): The number of output categories (classes) for classification.
    activation (str): The activation function to use in the output layer (e.g., 'softmax').

    Returns:
    keras.models.Sequential: A compiled CNN model ready for training.

    The model architecture:
    - 4 Convolutional layers with ReLU activation and Max Pooling
    - Flatten layer
    - Dense layer with 1024 units and ReLU activation
    - Dropout layer (20% dropout rate)
    - Output Dense layer with n_categories units and specified activation
    """
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_categories, activation=activation))
    return model