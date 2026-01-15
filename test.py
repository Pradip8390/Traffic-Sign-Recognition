import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Parameters
frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.75  # Probability threshold
font = cv2.FONT_HERSHEY_SIMPLEX

# Set up video capture
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# Load the trained model
model = load_model("model.h5")

# Functions
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def getCalssName(classNo):
    classNames = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h', 'Speed Limit 60 km/h',
        'Speed Limit 70 km/h', 'Speed Limit 80 km/h', 'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h',
        'Speed Limit 120 km/h', 'No passing', 'No passing for vehicles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles',
        'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left',
        'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road',
        'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
        'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits',
        'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left',
        'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing',
        'End of no passing by vehicles over 3.5 metric tons'
    ]
    return classNames[classNo]

while True:
    # Read image
    success, imgOrignal = cap.read()
    if not success:
        break
    
    # Process image
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    
    # Convert to 3 channels for display
    img_display = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.imshow("Processed Image", img_display)
    
    # Prepare image for prediction
    img = img.reshape(1, 32, 32, 1)
    
    # Predict
    predictions = model.predict(img)
    classIndex = np.argmax(predictions)
    probabilityValue = np.max(predictions)
    
    # Display result
    if probabilityValue > threshold:
        cv2.putText(imgOrignal, "CLASS: " + getCalssName(classIndex), (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, "PROBABILITY: " + str(round(probabilityValue * 100, 2)) + "%", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("Result", imgOrignal)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
