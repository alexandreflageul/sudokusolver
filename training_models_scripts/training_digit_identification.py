## DO NOT TOUCH THIS SECTION
# training model digits identification
from tensorflow.keras import layers, Sequential, optimizers, losses, metrics
def model_identify_digits(): # model d'identification des digits
    
    model = Sequential([
        layers.Conv2D(filters=10, kernel_size=3, activation="relu", input_shape=(28,28,1)),
        layers.Conv2D(filters=10, kernel_size=3, activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(filters=10, kernel_size=3, activation="relu"),
        layers.Conv2D(filters=10, kernel_size=3, activation="relu"),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(10, activation="softmax"),
        ])

    return model

# import data for training and validation
from tensorflow.keras.datasets import mnist
(X_identification_train, y_identification_train), (X_identification_test, y_identification_test) = mnist.load_data()
# NOTE: this dataset, are numpy arrays with high pixel value equal to numbers and low pixel valu equal to background

## some preprocessing of the data
identify_digits = model_identify_digits()

optimizer = optimizers.Adam()
loss = losses.SparseCategoricalCrossentropy()
metric = ["accuracy"]
    
identify_digits.compile(optimizer=optimizer, loss=loss, metrics=metric)

history = identify_digits.fit(X_identification_train, y_identification_train,
                              validation_data=(X_identification_test, y_identification_test),
                              epochs=10)

identify_digits.save("digits_identification_model.h5")

# Visualisation des valeurs du modele
# la visualisation des valeurs de l'évolution des valeurs de la fonction de cout renseigne sur l'overfitting
# (si l'écart entre les courbes est grand et que le train est meilleur que le test)
from plotly import graph_objects as go
import numpy as np
color_chart = ["#4B9AC7", "#4BE8E0", "#9DD4F3", "#97FBF6", "#2A7FAF", "#23B1AB", "#0E3449", "#015955"]

fig = go.Figure(data=[go.Scatter(y=history.history["loss"],
                                 name="Training loss",
                                 mode="lines",
                                 marker=dict(color=color_chart[0]) ),
                      go.Scatter(y=history.history["val_loss"],
                                 name="Validation loss",
                                 mode="lines",
                                 marker=dict(color=color_chart[1]) )
                                ])

# Add shapes
fig.add_shape(type="line",
              x0=np.argmin(history.history["val_loss"]),
              y0=np.min(history.history["loss"]),
              x1=np.argmin(history.history["val_loss"]),
              y1=np.max(history.history["loss"]),
              line=dict(color="red",width=3,dash="dot"),
              name="overfitting limit")

fig.update_layout(title='Training and val loss across epochs',
                  xaxis_title='epochs',
                  yaxis_title='Cross Entropy')
fig.show()
