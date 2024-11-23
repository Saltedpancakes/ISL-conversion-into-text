import numpy as np
import cv2
import keras
import tensorflow as tf
import serial  # Import the pyserial library
import time

# Initialize serial communication (Make sure you use the correct COM port for your Arduino)
ser = serial.Serial('COM6', 9600)  # Change 'COM3' to the appropriate port on your system
time.sleep(2)  # Wait for the serial communication to initialize

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    except RuntimeError as e:
        print(e)

# Load your pre-trained model
model = keras.models.load_model(r"C:\Users\laxmi\Indian-Sign-Language-Recognition\final-models\model-digit.h5")
cam = cv2.VideoCapture(0)

number_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (319, 9), (620 + 1, 309), (0, 255, 0), 1)
    roi = frame[10:300, 320:620]

    # Preprocess the image for the model
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gaussblur = cv2.GaussianBlur(gray, (5, 5), 2)
    smallthres = cv2.adaptiveThreshold(gaussblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2.8)
    ret, final_image = cv2.threshold(smallthres, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    final_image = cv2.resize(final_image, (128, 128))

    # Reshape the image and predict the digit
    final_image = np.reshape(final_image, (1, final_image.shape[0], final_image.shape[1], 1))
    pred = model.predict(final_image)
    predicted_digit = number_dict[np.argmax(pred)]

    # Print the predicted digit on the console
    print(predicted_digit)

    # Send the predicted digit to the Arduino over the serial port
    ser.write(predicted_digit.encode())  # Convert string to bytes and send

    # Display the prediction on the OpenCV window
    cv2.putText(frame, predicted_digit, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    cv2.imshow("Frame", frame)

    # Wait for the user to press the ESC key to exit
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# Release the camera and close the OpenCV window
cam.release()
cv2.destroyAllWindows()
ser.close()  # Close the serial communication
