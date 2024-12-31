import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, classification_report


image_size = (64, 64)
num_classes = 5 

# Load images from clustered folders
def load_images_from_clusters(base_dir):
    images = []
    labels = []
    
    for label in range(num_classes):
        cluster_dir = os.path.join(base_dir, f'cluster_{label}')
        for filename in os.listdir(cluster_dir):
            img_path = os.path.join(cluster_dir, filename)
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.resize(image, image_size)
                images.append(image)
                labels.append(label)
    
    return np.array(images), np.array(labels)

base_dir = './newtrain'  
X, y = load_images_from_clusters(base_dir)


X = X.astype('float32') / 255.0


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2,
                             height_shift_range=0.2, shear_range=0.2,
                             zoom_range=0.2, horizontal_flip=True,
                             fill_mode='nearest')


def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


model = create_cnn_model()
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10)


y_pred = np.argmax(model.predict(X_test), axis=-1)

# Calculate accuracy and precision
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(classification_report(y_test, y_pred))

model.save('model.h5')


