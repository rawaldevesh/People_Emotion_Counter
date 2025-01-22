from flask import Flask, render_template, Response
import cv2
from deepface import DeepFace

app = Flask(__name__)

# Initialize global counters
total_people_count = 0
happy_count = 0
not_happy_count = 0

def analyze_emotion(frame):
    """
    Analyze emotions in the given frame using DeepFace.
    """
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        # Extract dominant emotion
        dominant_emotion = result[0]['dominant_emotion']
        return dominant_emotion
    except Exception as e:
        print(f"Emotion detection error: {e}")
        return None

def generate_video_feed():
    """
    Generates video frames with people counting and emotion detection.
    """
    global total_people_count, happy_count, not_happy_count
    cap = cv2.VideoCapture(0)  # Use the default camera (camera index 0)
    
    if not cap.isOpened():
        raise Exception("Error: Could not access the camera. Ensure it's connected and free from other apps.")

    # Load pre-trained Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        total_people_count = len(faces)

        # Reset emotion counters for the frame
        happy_count = 0
        not_happy_count = 0

        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Emotion detection for each face
            face_roi = frame[y:y + h, x:x + w]
            emotion = analyze_emotion(face_roi)
            if emotion:
                color = (0, 255, 0) if emotion == 'happy' else (0, 0, 255)
                label = f"{emotion}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if emotion == 'happy':
                    happy_count += 1
                else:
                    not_happy_count += 1

        # Overlay counters on the video
        cv2.putText(frame, f"Total People: {total_people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Happy: {happy_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Not Happy: {not_happy_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as part of a multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    """
    Render the main page with the video feed.
    """
    return render_template('index.html')

@app.route('/start_camera')
def start_camera():
    """
    Start the camera feed for the video.
    """
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
