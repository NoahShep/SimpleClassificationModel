from tokenize import String
import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Rescaling
from tensorflow.keras.metrics import Precision, Recall, Accuracy
from tensorflow.keras.models import load_model

class ImageClassificationModel:
    """
    Initializes the classification model's variables for whenever you are 
    cleaning data, gathering data, fitting your model, exporting your model, 
    etc.

        * directory:  where your model's folder should be (within it should be 
          folders for model saves, testing data, fitting data, and logs)

        * num_of_classes:  the number of classes for your classification

        * color_mode:  how many color channels your model requires
          to differentiate between classifications (grayscale, rgb, or rgba).

        * batch_size:  how many training samples are worked until parameters
          are shifted within the model.

        * image_size:  the aspect ratio you would like your data images to be
          cropped/stretched to (written as (x,y))

        * shuffle:  whether your data will be randomized between classes.

        * seed:  used to keep consistency between within model fitting.

        * validation_split:  percentage of your data that you want
          to be used for validation of your model (float from 0 to 1).

        * crop_to_aspect_ratio:  whether the program will crop your images to
          the desired aspect ratio or stretch them.

    More information at https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
    """
    def __init__(
        self, 
        directory:String, 
        num_of_classes:int,
        color_mode:String='grayscale',
        batch_size:int=32,
        image_size=(256,256),
        shuffle:bool=True,
        seed:int=None,
        validation_split:float=0,
        crop_to_aspect_ratio:bool=False,
        test_split:float=0
    ) -> None:
        #   Initializes all of the model variables
        self.directory = directory
        self.num_of_classes = num_of_classes
        self.color_mode = color_mode
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.seed = seed
        self.validation_split = validation_split
        self.crop_to_aspect_ratio = crop_to_aspect_ratio
        self.test_split = test_split
        self.data = None
        self.test_data = None
        self.model = None

        #   Prevents out of memory errors
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    """
    Goes through your data to check if each image can be read by cv2 and has 
    a file extension that you desire. 
    """
    def clean_data(
        self,
        exts:list=['jpg','png','jpeg','bmp'],
        data_filename:String='data',
        prints:bool=True,
        delete:bool=False
    ):
        data_dir = os.path.join(self.directory, data_filename)
        #   Goes through each image in each image class to remove improper images
        for image_class in os.listdir(data_dir):
            #   Grabs each image
            for image in os.listdir(os.path.join(data_dir, image_class)):
                #   Gets each image path
                image_path = os.path.join(data_dir, image_class, image)
                try:
                    #   Tries to read the image using computer vision (cv2) and get its tip (.png, .jpg, etc)
                    img = cv2.imread(image_path)
                    ext = imghdr.what(image_path)

                    #   If the tip is not preferred, remove the file.
                    if ext not in exts:
                        if prints: print(f'Image not in ext list {image}')
                        if delete: os.remove(image_path)

                except Exception as e:
                    #   If the image has some issue, remove the file.
                    if prints: print(f'Issue with image {image}')
                    if delete: os.remove(image_path)
    
    """
    Uses your data folder within your directory folder to create a TensorFlow
    Dataset object automatically, and then adjusts it to help with the training
    (unless specified otherwise within the parameters).
    """
    def load_data(
        self,
        data_filename:String='data'
    ):
        #   Creates an automatic dataset based on your data folder within your directory
        self.data = tf.keras.utils.image_dataset_from_directory(
            directory=os.path.join(self.directory, data_filename),
            color_mode=self.color_mode,
            batch_size=self.batch_size,
            image_size=self.image_size,
            shuffle=self.shuffle,
            seed=self.seed,
            crop_to_aspect_ratio=self.crop_to_aspect_ratio
        )

    """
    Creates a general classification model using 2D convolutional layers in with
    2D max pooling and non-linear activation functions in order to find features 
    and patterns within the data images.

    The end of the model consists of dense layers that interpret what the
    convolutional layers output and finalize them into outputs that classify the
    images between 0 and 1 for each output node.

    The default loss function is for models with multiple potential outputs. Use
    'tf.losses.BinaryCrossentropy()' for your loss function if you have only two
    classifications you want to test for.
    """
    def create(
        self,
        input_optimizer:String='adam',
        input_loss=None,
        input_metrics:list=['accuracy']
    ):
        if input_loss == None:
            if self.num_of_classes <= 2:
                input_loss = tf.losses.BinaryCrossentropy()
            else:
                input_loss = tf.losses.SparseCategoricalCrossentropy()

        #   Creating the model and adding layers
        self.model = Sequential()

        #   Check the number of color channels
        color_mode = self.color_mode.lower()
        num_of_color_channels = 1
        if color_mode == 'rgb':
            num_of_color_channels = 3
        elif color_mode == 'rgba':
            num_of_color_channels = 4

        #   INPUT LAYER
        self.model.add(Rescaling(1./255, input_shape=(self.image_size[0], self.image_size[1], num_of_color_channels)))
        self.model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(self.image_size[0], self.image_size[1], num_of_color_channels)))
        self.model.add(MaxPooling2D())
        #   HIDDEN LAYER 1
        self.model.add(Conv2D(32, (3,3), 1, activation='relu'))
        self.model.add(MaxPooling2D())
        #   HIDDEN LAYER 2
        self.model.add(Conv2D(16, (3,3), 1, activation='relu'))
        self.model.add(MaxPooling2D())
        #   HIDDEN LAYER 3
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        #   OUTPUT LAYER
        self.model.add(Dense(self.num_of_classes, activation='sigmoid'))

        # Compile the model with an optimizer, loss function, and metrics to measure
        self.model.compile(
            optimizer=input_optimizer,
            loss=input_loss, 
            metrics=input_metrics
        )

    """
    Trains the model using the model's fit method along with tensorboard to 
    keep track of how the model's accuracy and losses change over time. 

    Uses the validation_split variable to determine how much data is for 
    training and how much is for validation while fitting.
        * If you have a low amount of data, using validation_split can cause
          errors if your batch_size is large.
    """
    def train(
        self,
        logs_filename:String='logs',
        epochs:int=5,
        show_analytics:bool=True
    ):
        #   Uses the logs folder as a way to log a model as it trains.
        log_dir = os.path.join(self.directory, logs_filename)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

        #   Gather the amount of training/validation data that should be used when fitting.
        train_size = int(len(self.data) * (1 - self.validation_split - self.test_split))
        validate_size = int(len(self.data) * self.validation_split)
        test_size = int(len(self.data) * self.test_split)
        #   Makes sure that all data fits into either training or validating.
        if train_size + validate_size + test_size < len(self.data):
            validate_size = len(self.data) - train_size - test_size

        #   Getting the training and validation data.
        training_data = self.data.take(train_size)
        validation_data = self.data.skip(train_size).take(validate_size)
        self.test_data = self.data.skip(train_size).skip(validate_size).take(test_size)

        #   Fits the model using the training data and validation data with 20 iterations.
        #   Storing it in a hist (history) variables allows us to look at how our model trained.
        hist = self.model.fit(training_data, epochs=epochs, validation_data=validation_data, callbacks=[tensorboard_callback])

        if not show_analytics: return

        #   Using matplotlib to show how the losses changed over time.
        fig = plt.figure()
        plt.plot(hist.history['loss'], color='teal', label='loss')
        if self.validation_split is not None: plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
        fig.suptitle('Loss', fontsize=20)
        plt.legend(loc='upper left')
        plt.show()

        #   Using matplotlib to show how the accuracies changed over time.
        fig = plt.figure()
        plt.plot(hist.history['accuracy'], color='teal', label='loss')
        if self.validation_split is not None: plt.plot(hist.history['val_accuracy'], color='orange', label='val_loss')
        fig.suptitle('Accuracy', fontsize=20)
        plt.legend(loc='upper left')
        plt.show()

    def get_test(
        self,
        test_filename:String='test'
    ):
        test_path = os.path.join(self.directory, test_filename)
        self.test_data = tf.keras.utils.image_dataset_from_directory(
            directory=test_path,
            color_mode=self.color_mode,
            batch_size=1,
            image_size=self.image_size,
            shuffle=False,
            crop_to_aspect_ratio=self.crop_to_aspect_ratio
        )

    """
    Uses an user-generated test folder as means for testing classifications with
    a model (the indices of each folder within the test folder is used as the
    expected output value).
    """
    def test(
        self,
        individual_prints:bool=True
    ):
        if (self.test_data is None):
            print('Test data has not been found yet. Run get_test() to use test files, or set test_split greater than 0 and run train().')
            return

        #   Creating instances of these classes to measure prediction.
        pre = Precision()
        re = Recall()
        acc = Accuracy()

        #   Goes through each batch to check measurements of performance
        for batch in self.test_data.as_numpy_iterator():
            #   Grabs the images and labels of each batch.
            X, y = batch
            #   Makes a prediction using the images as to what the labels will be
            y_actual = self.model.predict(X)
            #   Find index of highest value (should be same as y) 
            y_actual = tf.constant(tf.math.argmax(y_actual[0]), shape=(1))
            #   Update measurements
            if individual_prints: print(f'Expected output: {y}, actual output: {y_actual}')
            pre.update_state(y, y_actual)
            re.update_state(y, y_actual)
            acc.update_state(y, y_actual)

        #   Prints out the measurements (1 is the best value to have).
        print(f'Precision: {pre.result()}, Recall: {re.result()}, Accuracy: {acc.result()}')

    """
    Saves the model within a '.h5' file within the model saves folder within 
    your directory ('.h5' is the extension used because HDF5 handles large,
    complex data well).
    """
    def save(
        self,
        models_filename:String='models',
        model_name:String='classification_model'
    ):
        models_path = os.path.join(self.directory, models_filename)
        self.model.save(os.path.join(models_path,f'{model_name}.h5'))

    """
    Loads your desired model using TensorFlow's load_model method.
    """
    def load(
        self,
        model_name:String='classification_model',
        models_filename:String='models'
    ): 
        models_path = os.path.join(self.directory, models_filename)
        try:
            self.model = load_model(os.path.join(models_path, f'{model_name}.h5'))
            print(f'Successfully loaded model {model_name}.h5')
        except Exception as e:
            print(f'Unable to load model {model_name}.h5: {e}')
