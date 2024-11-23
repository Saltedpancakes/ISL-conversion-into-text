import serial
import time

# Set up the serial connection (replace 'COM5' with your actual port)
ser = serial.Serial('COM5', 9600)  # Adjust baud rate and COM port as needed
time.sleep(2)  # Allow time for the connection to initialize

# Attempt to read the digit from the text file
try:
    with open("C:/Users/laxmi/Indian-Sign-Language-Recognition/output.txt", "r") as file:
        prediction = file.read().strip()
except PermissionError:
    print("PermissionError: Could not access the file.")
    ser.close()
    exit(1)  # Exit the script if there's a permission error

# Send the digit to the microcontroller
ser.write(prediction.encode())

# Close the serial connection
ser.close()
