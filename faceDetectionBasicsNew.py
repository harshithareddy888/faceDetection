import cv2
import mediapipe as mp

# ✅ Replace with any actual image on your PC
image_path = r"C:\Users\harshitha reddy\OneDrive\Pictures\Camera Roll\WIN_20240714_18_56_23_Pro.jpg"  # <-- put a real image path here

img = cv2.imread(image_path)
if img is None:
    print(f"❌ Could not load image at {image_path}")
    exit()

# Convert to RGB for MediaPipe
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Setup MediaPipe Face Detection
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Process the image
results = faceDetection.process(imgRGB)
print("✅ Results:", results)

# Draw detections if any
if results.detections:
    for detection in results.detections:
        mpDraw.draw_detection(img, detection)
        print("Detection confidence:", detection.score[0])

# Show the output
cv2.imshow("Face Detection Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
