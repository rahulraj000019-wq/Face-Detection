Face, Eye, Nose & Mouth Detection using OpenCV
This project is a real-time face and facial feature detection system built using Python and OpenCV.
It uses Haar Cascade classifiers to detect a person’s face, eyes, nose, and mouth through a webcam feed.
The project is beginner-friendly and a great starting point for anyone interested in Computer Vision and AI-based image processing.
Features
Real-time webcam detection
Detects:
Face
Eyes
Nose
Mouth
Draws labeled bounding boxes on detected features
Uses lightweight and fast Haar Cascade models
Technologies Used
Python
OpenCV (cv2)
Haar Cascade Classifiers
📂 Project Structure
├── main.py
├── haarcascade_frontalface_default.xml
├── haarcascade_eye.xml
├── haarcascade_mcs_nose.xml
├── haarcascade_mcs_mouth.xml
└── README.md
Make sure all Haar Cascade .xml files are in the same directory as the Python file.
How It Works
The webcam captures live video frames.
Each frame is converted to grayscale.
Haar Cascade classifiers scan the frame to detect:
Face first
Then eyes, nose, and mouth inside the face region
Rectangles and labels are drawn around detected features.
Press q to exit the program.
How to Run the Project
Install Required Libraries
pip install opencv-python
Clone the Repository
Run the Script
python main.py
Output
Blue box → Face
Green box → Eyes
Pink box → Nose
Red box → Mouth
Detection happens live through your webcam.
Common Issues
Webcam not opening
Check camera permissions or change VideoCapture(0) to VideoCapture(1)
Cascades not loading
→ Ensure .xml files are present in the correct folder
Future Improvements
Improve accuracy using deep learning models
Add face recognition
Save detected images automatically
Convert into a GUI application
Contributing
Contributions are welcome!
Feel free to fork the repository and submit a pull request.
License
This project is open-source and free to use for learning and educational purposes.
