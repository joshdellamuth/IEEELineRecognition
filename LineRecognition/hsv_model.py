import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import cv2
import os

# Load images from folders
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = load_and_preprocess_image(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
    return np.array(images)

# Load and preprocess images
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224)) # Resize to the input size expected by the model
    return image / 255.0 # Normalize pixel values

image_folder_path = "line_images"
# Prepare the dataset
image_dataset = load_images_from_folder(image_folder_path)

hsv_values = [
    [10,36,20,100,120,210], #Image 1
    [10,36,20,100,120,210], #Image 2
    [10,36,20,122,120,231], #Image 3
    [10,36,20,140,120,240], #Image 4
    [10,36,20,100,120,210], #Image 5
    [10,36,20,100,120,220], #Image 6
    [10,36,20,100,120,210], #Image 7
    [10,36,8,100,120,210], #Image 8
    [10,36,8,100,120,210], #Image 9
    [10,36,8,100,120,210], #Image 10
    [10,36,20,130,170,236], #Image 13
    [10,36,0,100,120,210], #Image 14
    [10,36,20,144,120,250], #Image 15
    [0,87,0,94,120,211], #Image 16
    [10,36,20,130,120,255], #Image 17
    [0,36,0,101,140,247], #Image 18
    [10,36,20,100,120,226], #Image 19
    [10,36,20,100,120,220], #Image 20
    [10,36,20,123,120,239], #Image 21
    [10,36,20,137,120,235], #Image 22
    [10,36,20,144,120,244], #Image 23
    [10,36,20,138,120,249], #Image 24
    [10,36,20,140,120,255] #Image 25
]
# Assuming hsv_values is a list of lists with 6 values each
hsv_dataset = np.array(hsv_values)

# Split dataset into training and testing
split = int(0.8 * len(image_dataset))
train_images, test_images = image_dataset[:split], image_dataset[split:]
train_hsv, test_hsv = hsv_dataset[:split], hsv_dataset[split:]

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(6) # Output layer with 6 nodes (lower and upper bounds for H, S, and V)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Configure the EarlyStopping callback
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20, verbose=1, restore_best_weights=True)

# Train the model with the callback
model.fit(train_images, train_hsv, epochs=250, validation_data=(test_images, test_hsv), callbacks=[early_stopping_callback])

validation_image_path = "testLine.jpg"

validation_image = load_and_preprocess_image(validation_image_path)
validation_image = np.expand_dims(validation_image, axis=0)
predicted_hsv = model.predict(validation_image)
predicted_hsv = predicted_hsv[0].astype(int) 

# Print or process the predicted HSV values


def update_yellow_detection(x):
    # Get the current positions of the trackbars
    low_h = cv2.getTrackbarPos('Low H', 'Yellow Detection')
    low_s = cv2.getTrackbarPos('Low S', 'Yellow Detection')
    low_v = cv2.getTrackbarPos('Low V', 'Yellow Detection')
    high_h = cv2.getTrackbarPos('High H', 'Yellow Detection')
    high_s = cv2.getTrackbarPos('High S', 'Yellow Detection')
    high_v = cv2.getTrackbarPos('High V', 'Yellow Detection')

    # Create the lower and upper bounds for the yellow color
    lower_yellow = np.array([low_h, low_s, low_v])
    upper_yellow = np.array([high_h, high_s, high_v])

    # Create a mask to isolate yellow regions
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Apply median blur to reduce noise
    blurred_mask = cv2.medianBlur(mask, 5)

    # Display the result
    cv2.imshow('Yellow Detection', blurred_mask)



# Read the image
image = cv2.imread(validation_image_path)

#IMAGE ABOVE DUMMY

if image is None:
    print("Could not read the image.")
    exit()

# Convert the image to the HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV', hsv)

# Create a window for the trackbars
cv2.namedWindow('Yellow Detection')


# Create trackbars for color change
cv2.createTrackbar('Low H', 'Yellow Detection', predicted_hsv[0], 179, update_yellow_detection)
cv2.createTrackbar('High H', 'Yellow Detection', predicted_hsv[1], 179, update_yellow_detection)
cv2.createTrackbar('Low S', 'Yellow Detection', predicted_hsv[2], 255, update_yellow_detection)
cv2.createTrackbar('High S', 'Yellow Detection', predicted_hsv[3], 255, update_yellow_detection)
cv2.createTrackbar('Low V', 'Yellow Detection', predicted_hsv[4], 255, update_yellow_detection)
cv2.createTrackbar('High V', 'Yellow Detection', predicted_hsv[5], 255, update_yellow_detection)

# Initial detection
update_yellow_detection(0)
print(predicted_hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()



