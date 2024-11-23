import numpy as np
import cv2
import keras
import tensorflow as tf

# Configure GPU settings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], 
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    except RuntimeError as e:
        print(e)

# Load the trained model
model = keras.models.load_model(r"C:\Users\laxmi\Indian-Sign-Language-Recognition\final-models\model-digit.h5")

# Try to open the camera
cam = cv2.VideoCapture(0)  # Change index if necessary

if not cam.isOpened():
    print("Could not open camera.")
else:
    print("Camera opened successfully.")
    number_dict = {0:'Zero', 1:'One', 2:'Two', 3:'Three', 4:'Four', 
                   5:'Five', 6:'Six', 7:'Seven', 8:'Eight', 9:'Nine'}

    while True:
        ret, frame = cam.read()  # Read frame from the camera
        if not ret:
            print("Failed to grab frame.")
            break  # Exit loop if frame not grabbed

        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (319, 9), (620 + 1, 309), (0, 255, 0), 1)
        roi = frame[10:300, 320:620]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gaussblur = cv2.GaussianBlur(gray, (5, 5), 2)
        smallthres = cv2.adaptiveThreshold(gaussblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2.8)
        ret, final_image = cv2.threshold(smallthres, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imshow("BW", final_image)
        final_image = cv2.resize(final_image, (128, 128))

        final_image = np.reshape(final_image, (1, final_image.shape[0], final_image.shape[1], 1))
        pred = model.predict(final_image)
        print(number_dict[np.argmax(pred)])
        cv2.putText(frame, number_dict[np.argmax(pred)], (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.imshow("Frame", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Break on 'ESC' key
            break

    # Release the camera and destroy all windows
    cam.release()
    cv2.destroyAllWindows()
