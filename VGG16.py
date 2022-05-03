# Import packages required

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Import the train and test splits of the CIFAR10 set.
# As the splits appear to be not randomized, I will leave them alone.
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# Normalize pixel intensities from the normal integer range of 0 to 255 to a scale of 0 to 1
X_train, X_test = X_train / 255, X_test / 255

# Check training and test sets
print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)

# As shown in the notebook file on collab, display the first 25 images in the training split
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i])
    plt.xlabel(class_names[y_train[i][0]])
plt.show()

# Check image shape to get size for input
print(X_test[0].shape)

# Initialize the VGG16 model as a sequential model
VGG16 = models.Sequential()

# First block with 2 convolutional layers
VGG16.add(layers.Conv2D(input_shape=(32, 32, 3), kernel_size=(3, 3), filters=64, padding="same", activation="relu"))
VGG16.add(layers.Conv2D(kernel_size=(3, 3), filters=64, padding="same", activation="relu"))

# Pooling step
VGG16.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Second block with 2 convolutional layers
VGG16.add(layers.Conv2D(kernel_size=(3, 3), filters=128, padding="same", activation="relu"))
VGG16.add(layers.Conv2D(kernel_size=(3, 3), filters=128, padding="same", activation="relu"))

# Pooling step
VGG16.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Third block with 3 convolutional layers and pooling
VGG16.add(layers.Conv2D(kernel_size=(3, 3), filters=256, padding="same", activation="relu"))
VGG16.add(layers.Conv2D(kernel_size=(3, 3), filters=256, padding="same", activation="relu"))
VGG16.add(layers.Conv2D(kernel_size=(3, 3), filters=256, padding="same", activation="relu"))

# Pooling step
VGG16.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Fourth block with 3 convolutional layers and pooling
VGG16.add(layers.Conv2D(kernel_size=(3, 3), filters=512, padding="same", activation="relu"))
VGG16.add(layers.Conv2D(kernel_size=(3, 3), filters=512, padding="same", activation="relu"))
VGG16.add(layers.Conv2D(kernel_size=(3, 3), filters=512, padding="same", activation="relu"))

# Pooling step
VGG16.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Fifth block with 3 convolutional layers and pooling
VGG16.add(layers.Conv2D(kernel_size=(3, 3), filters=512, padding="same", activation="relu"))
VGG16.add(layers.Conv2D(kernel_size=(3, 3), filters=512, padding="same", activation="relu"))
VGG16.add(layers.Conv2D(kernel_size=(3, 3), filters=512, padding="same", activation="relu"))

# Pooling step
VGG16.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Final sixth block with fully connected layers

# Flatten into a vector
VGG16.add(layers.Flatten())
# First two FC layers
VGG16.add(layers.Dense(units=4096, activation="relu"))
VGG16.add(layers.Dense(units=4096, activation="relu"))
# Final FC layer, softmax for generalizing the output on a sigmoid function.
VGG16.add(layers.Dense(units=1000, activation="softmax"))

# Get a summary to make sure I have the parameters correct
VGG16.summary()

# Make variables for easy adjustment of epochs, batchsize and LR
n_epochs = 12
n_batch_size = 24
#n_learning_rate = 0.005

# Compile the model, I am using Adam as the optimizer and loss is derived from the cross entropy function
VGG16.compile(optimizer="SGD",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the VGG16 model
trained_VGG16 = VGG16.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=n_epochs, batch_size=n_batch_size)

# Print accuracy for part 3 of assignment
loss, accuracy = VGG16.evaluate(X_test, y_test, verbose=2)
print(accuracy)
print(loss)


# Make plots for part 2
# Note that I was not sure if this is supposed to be just training accuracy / loss of if it includes validation also
# so I included validation accuracy / loss as well

# Get the training accuracy and validation accuracy for the graph
training_accuracy = trained_VGG16.history['accuracy']
validation_accuracy = trained_VGG16.history['val_accuracy']

# Make the plot for accuracy over each epoch
plt.plot(training_accuracy)
plt.plot(validation_accuracy)
plt.title('Accuracy With Regard to Epochs')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy Percentage')
plt.xticks(range(n_epochs))
plt.show()

# Get the training loss and validation loss for the graph
training_loss = trained_VGG16.history['loss']
validation_loss = trained_VGG16.history['val_loss']

# Make the plot for loss over each epoch
plt.plot(training_loss)
plt.plot(validation_loss)
plt.title('Loss With Regard to Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss Percentage')
plt.xticks(range(n_epochs))
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()
