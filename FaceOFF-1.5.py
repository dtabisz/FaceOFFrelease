import cv2
import numpy as np

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(gray_frame):
    return face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

def update_emergence(face_circles, emergence_dict):
    new_emergence = {}
    for (_, center, radius) in face_circles:
        matched = False
        for prev_center, prev_value in emergence_dict.items():
            dist = np.linalg.norm(np.array(center) - np.array(prev_center))
            if dist < radius:
                new_value = min(prev_value + 2, 11)
                new_emergence[center] = new_value
                matched = True
                break
        if not matched:
            new_emergence[center] = 2

    for key in new_emergence:
        new_emergence[key] = max(new_emergence[key] - 1, 0)
    return new_emergence

def detect_nearest_face(face_circles):
    max_radius = 0
    nearest_face = None
    for (_, center, radius) in face_circles:
        if radius > max_radius:
            max_radius = radius
            nearest_face = (center, radius)
    return nearest_face

def overlay_face(frame, nearest_face, face_center, face_radius, emergence_value):
    if nearest_face:
        (nx, ny), nr = nearest_face
        mask = np.zeros_like(frame)
        cv2.circle(mask, (nx, ny), nr, (255, 255, 255), -1)
        nearest_face_img = cv2.bitwise_and(frame, mask)
        nearest_crop = nearest_face_img[ny - nr:ny + nr, nx - nr:nx + nr]
        
        try:
            resized_nearest_face = cv2.resize(nearest_crop, (face_radius*2, face_radius*2))
            alpha_mask = np.zeros((face_radius*2, face_radius*2), dtype=np.float32)
            
            # Apply gradient transparency based on distance from center
            for i in range(face_radius*2):
                for j in range(face_radius*2):
                    dist = np.sqrt((i - face_radius) ** 2 + (j - face_radius) ** 2)
                    max_dist = face_radius
                    alpha = max(0, 1 - (dist / max_dist))
                    alpha *= emergence_value / 10.0
                    alpha_mask[i, j] = alpha
                    
            overlay = resized_nearest_face.astype(float)
            roi = frame[face_center[1] - face_radius: face_center[1] + face_radius, face_center[0] - face_radius: face_center[0] + face_radius].astype(float)
            for c in range(3):
                roi[:, :, c] = roi[:, :, c] * (1 - alpha_mask) + overlay[:, :, c] * alpha_mask
            frame[face_center[1] - face_radius: face_center[1] + face_radius, face_center[0] - face_radius: face_center[0] + face_radius] = roi.astype(np.uint8)
        except Exception as e:
            print(f"Error in overlay: {e}")
    return frame

# Function to add noise to the image
def add_noise(frame, noise_level):
    row, col, ch = frame.shape
    mean = 0
    sigma = noise_level * 20  # Adjust the noise intensity (higher value for more noise)
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy_image = np.uint8(np.clip(frame + gauss, 0, 255))
    return noisy_image

# Function to display the help text
def display_help(frame, show_circles, noise_level, is_bw):
    help_text = [
        "FaceOFF 1.5",
        "Is it 'we' that we are? Or are we just a collective projection of others?",
        "Do we see another person, or just our own image?",
        "",
        "",
        "",
        "H - Toggle help display",
        "R - Show circles: " + ("ON" if show_circles else "OFF"),
        "The number next to the circle indicates how many frames the face has been detected for.",
        "W - Toggle black and white: " + ("ON" if is_bw else "OFF"),
        f"Noise level: {noise_level}",
        "1-5 - Change noise level (1: None, 5: Max)",
        "Q - Quit",
        "",
        "copyleft terms ;)",
        "FaceOFF can be used freely and redistributed under GPLv3 License",
        "Cooperation DT3, QwQ (Qwen Team), GPT4 (OpenAI)"
    ]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    y0, dy = 30, 40  # Start position of the text

    for i, line in enumerate(help_text):
        cv2.putText(frame, line, (10, y0 + i * dy), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

def main():
    width, height = 2560 , 1600 #OTPUT Resolution
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    show_circles = True
    emergence_dict = {}
    noise_level = 0  # No noise initially
    is_bw = False  # Start in color mode
    show_help = False  # Help display is off initially

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Add noise to the image
        frame = add_noise(frame, noise_level)

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)
        
        face_circles = []
        for (x, y, w, h) in faces:
            center = (x + w // 2, y + h // 2)
            radius = h // 2
            face_circles.append(((x, y, w, h), center, radius))

        emergence_dict = update_emergence(face_circles, emergence_dict)

        nearest_face = detect_nearest_face(face_circles)

        for (_, center, radius) in face_circles:
            color = (0, 0, 255) if (center, radius) == nearest_face else (0, 255, 0)
            if show_circles:
                cv2.circle(frame, center, radius, color, 2)
                cv2.putText(frame, f'{emergence_dict.get(center, 0)}', (center[0] - 10, center[1] - radius - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            frame = overlay_face(frame, nearest_face, center, radius, emergence_dict.get(center, 0))

        frame_resized = cv2.resize(frame, (width, height))

        # Convert to grayscale for black-and-white display if 'W' is pressed
        if is_bw:
            frame_bw = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            frame_resized = cv2.cvtColor(frame_bw, cv2.COLOR_GRAY2BGR)

        # Show help text if enabled
        if show_help:
            display_help(frame_resized, show_circles, noise_level, is_bw)

        cv2.imshow('Face Detection with Nearest Face Overlay', frame_resized)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            show_circles = not show_circles
        elif key == ord('w'):  # Toggle black-and-white mode with 'W'
            is_bw = not is_bw
        elif key == ord('1'):  # Set noise level to 0
            noise_level = 0
        elif key == ord('2'):  # Set noise level to 1
            noise_level = 1
        elif key == ord('3'):  # Set noise level to 2
            noise_level = 2
        elif key == ord('4'):  # Set noise level to 3
            noise_level = 3
        elif key == ord('5'):  # Set noise level to 4
            noise_level = 4
        elif key == ord('h'):  # Toggle help with 'H'
            show_help = not show_help

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
