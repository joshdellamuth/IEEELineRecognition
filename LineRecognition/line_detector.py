import cv2
import numpy as np

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
image = cv2.imread('testLine3.jpg')

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
cv2.createTrackbar('Low H', 'Yellow Detection', 10, 179, update_yellow_detection)
cv2.createTrackbar('High H', 'Yellow Detection', 36, 179, update_yellow_detection)
cv2.createTrackbar('Low S', 'Yellow Detection', 20, 255, update_yellow_detection)
cv2.createTrackbar('High S', 'Yellow Detection', 100, 255, update_yellow_detection)
cv2.createTrackbar('Low V', 'Yellow Detection', 120, 255, update_yellow_detection)
cv2.createTrackbar('High V', 'Yellow Detection', 210, 255, update_yellow_detection)

# Initial detection
update_yellow_detection(0)

cv2.waitKey(0)
cv2.destroyAllWindows()


