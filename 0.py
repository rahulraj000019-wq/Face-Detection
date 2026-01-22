import cv2

# Load Haar cascade classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

# Check if the cascades are loaded successfully
if face_cascade.empty() or eye_cascade.empty() or nose_cascade.empty() or mouth_cascade.empty():
    print("Error loading cascades!")
    exit()

# Start video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, img = video_capture.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Define region of interest for eyes, nose, and mouth detection
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            cv2.putText(roi_color, "Eye", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Detect nose
        noses = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        if len(noses) > 0:
            nx, ny, nw, nh = noses[0]
            cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 255), 2)
            cv2.putText(roi_color, "Nose", (nx, ny - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        # Detect mouth
        mouth_roi_gray = roi_gray[int(h / 2):, :]
        mouths = mouth_cascade.detectMultiScale(mouth_roi_gray, scaleFactor=1.5, minNeighbors=5)
        if len(mouths) > 0:
            mx, my, mw, mh = mouths[0]
            cv2.rectangle(roi_color, (mx, int(h / 2) + my), (mx + mw, int(h / 2) + my + mh), (0, 0, 255), 2)
            cv2.putText(roi_color, "Mouth", (mx, int(h / 2) + my - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Face, Eyes, Nose, and Mouth Detection', img)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()


