import cv2
import dlib
import numpy as np

# Inicjalizacja detektorów
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Plik .dat do pobrania
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Globalne przełączniki
show_rectangles = True
noise_level = 0  # Poziom szumu: 0 (brak) do 8 (maksymalny)
grayscale_mode = False  # Przełącznik trybu czarno-białego
show_help = False  # Przełącznik instrukcji
language = "pl"  # Domyślny język: polski

# Docelowe rozmiary wyświetlania
TARGET_WIDTH = 2560
TARGET_HEIGHT = 1600

def add_noise(frame, noise_level):
    """Nakłada szum na obraz."""
    if noise_level == 0:
        return frame
    intensity = noise_level * 3  # Zmniejszona intensywność szumu
    noise = np.random.normal(0, intensity, frame.shape).astype(np.int16)
    noisy_frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy_frame

def detect_faces(frame):
    """Wykrywa twarze za pomocą dlib i haar cascade."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = []

    # Dlib detekcja
    faces_dlib = detector(gray)
    for face in faces_dlib:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        try:
            shape = predictor(gray, face)  # Landmarks
            landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
            faces.append({'method': 'dlib', 'x': x, 'y': y, 'w': w, 'h': h, 'center': (x + w // 2, y + h // 2), 'landmarks': landmarks})
        except Exception as e:
            print(f"Warning: Could not detect landmarks: {e}")

    # Haar detekcja (tylko jeśli Dlib nie wykrył twarzy)
    faces_haar = haar_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))
    for (x, y, w, h) in faces_haar:
        overlap = any(abs(x - face['x']) < face['w'] and abs(y - face['y']) < face['h'] for face in faces)
        if not overlap:
            faces.append({'method': 'haar', 'x': x, 'y': y, 'w': w, 'h': h, 'center': (x + w // 2, y + h // 2)})

    return faces

def overlay_faces(frame, faces):
    """Nakłada twarze na inne twarze z gradientem."""
    if not faces:
        return frame

    # Wybór największej twarzy
    max_face = max(faces, key=lambda f: f['h'])

    for face in faces:
        if face == max_face:
            continue  # Nie nadpisujemy największej twarzy

        target_x, target_y, target_w, target_h = face['x'], face['y'], face['w'], face['h']
        source_x, source_y, source_w, source_h = max_face['x'], max_face['y'], max_face['w'], max_face['h']

        # Ograniczenie współrzędnych źródłowych do granic obrazu
        source_x = max(0, source_x)
        source_y = max(0, source_y)
        source_w = min(source_w, frame.shape[1] - source_x)
        source_h = min(source_h, frame.shape[0] - source_y)

        if source_w <= 0 or source_h <= 0:  # Sprawdzanie pustego regionu
            continue

        # Wycięcie i skalowanie twarzy
        source_face = frame[source_y:source_y + source_h, source_x:source_x + source_w]
        resized_face = cv2.resize(source_face, (target_w, target_h))

        # Tworzenie maski gradientowej
        mask = np.zeros((target_h, target_w), dtype=np.float32)
        cv2.circle(mask, (target_w // 2, target_h // 2), target_h // 2, 1, -1)
        mask = cv2.GaussianBlur(mask, (15, 15), 10)
        mask = np.dstack([mask] * 3)  # Dla trzech kanałów

        # Nakładanie twarzy z gradientem
        target_area = frame[target_y:target_y + target_h, target_x:target_x + target_w]
        blended = resized_face * mask + target_area * (1 - mask)
        frame[target_y:target_y + target_h, target_x:target_x + target_w] = blended.astype(np.uint8)

    return frame

def draw_help(frame):
    """Wyświetla instrukcje na ekranie."""
    if language == "pl":
        instructions = [
            "FaceOFF - czy doznajesz innych", 
            "czy poprzez innych doznajesz siebie",
            "",
            "Instrukcja:",
            "Q - Wyjscie",
            "L - PL/EN",
            "R - Pokaz/ukryj ramki na twarzach",
            "W - Przelacz kolor/czarno-bialy",
            "1-9 - Ustaw poziom szumu",
            "H - Pokaz/ukryj instrukcje",
            "",
             "FaceOFF podlega licencji GPLv3",
            "Kooperacja: DT3, GPT3, GPT4"
        ]
    else:  # language == "en"
        instructions = [
            "FaceOFF - do you experience others", 
            "or through others do you experience yourself",
            "",
            "Instructions:",
            "Q - Exit",
            "L - language",
            "R - Show/Hide face rectangles",
            "W - Toggle color/black and white",
            "1-9 - Set noise level",
            "H - Show/Hide instructions",
            "",
            "FaceOFF is licensed under GPLv3",
            "Cooperation: DT3, GPT3, GPT4"
        ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    line_height = 20

    y = 30
    for line in instructions:
        cv2.putText(frame, line, (30, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_height

def main():
    global show_rectangles, noise_level, grayscale_mode, show_help, language
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Face Overlay", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Face Overlay", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = add_noise(frame, noise_level)

        # Przełączanie trybu czarno-białego
        if grayscale_mode:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        faces = detect_faces(frame)

        if show_rectangles:
            # Rysowanie ramek
            for face in faces:
                color = (0, 255, 0) if face['method'] == 'dlib' else (255, 0, 0)
                cv2.rectangle(frame, (face['x'], face['y']), (face['x'] + face['w'], face['y'] + face['h']), color, 2)

        frame = overlay_faces(frame, faces)

        # Wyświetlanie instrukcji
        if show_help:
            draw_help(frame)

        # Skalowanie obrazu do docelowej rozdzielczości
        frame_resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

        cv2.imshow("Face Overlay", frame_resized)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Wyjście
            break
        elif key == ord('r'):  # Przełączanie ramek
            show_rectangles = not show_rectangles
        elif ord('1') <= key <= ord('9'):  # Regulacja poziomu szumu
            noise_level = key - ord('1')  # Ustawienie poziomu szumu (0-8)
        elif key == ord('w'):  # Przełączanie trybu czarno-białego
            grayscale_mode = not grayscale_mode
        elif key == ord('h'):  # Przełączanie wyświetlania instrukcji
            show_help = not show_help
        elif key == ord('l'):  # Przełączanie języka
            language = "pl" if language == "en" else "en"  # Zmiana języka

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
