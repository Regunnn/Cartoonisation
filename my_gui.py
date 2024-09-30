import cv2
import numpy as np
from tkinter.filedialog import *

x = int(input('ENTER 1.Cartoonisation or 2.Sketching : '))

if x == 1:
    photo = askopenfilename()
    img = cv2.imread(photo)

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey = cv2.medianBlur(grey, 5)
    edges = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    # Cartoonize
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    cv2.imshow("Image", img)
    cv2.imshow("Cartoon", cartoon)

    # Save
    cv2.imwrite("cartoon.jpg", cartoon)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
elif x == 2:
    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Getting width and height of the image
        height, width, _ = frame.shape

        # Creating a copy of the image using resize function
        resizedImage = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        # Creating a 3x3 kernel to sharpen the image/frame
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        # Applying the kernel on the frame using filter2D function.
        sharpenImage = cv2.filter2D(resizedImage, -1, kernel)

        # Converting the image into a grayscale image.
        gray = cv2.cvtColor(sharpenImage, cv2.COLOR_BGR2GRAY)

        # Creating the inverse of the sharpen image.
        inverseImage = 255 - gray

        # Applying Gaussian Blur on the image.
        bluredImage = cv2.GaussianBlur(inverseImage, (15, 15), 0, 0)

        # Create a pencil sketch using the divide function in OpenCV.
        pencilSketch = cv2.divide(gray, 255 - bluredImage, scale=256)

        cv2.imshow('Sharpen Image', sharpenImage)
        cv2.imshow("pencilSketch", pencilSketch)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    camera.release()
