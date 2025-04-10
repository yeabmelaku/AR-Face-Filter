import cv2
import numpy as np
import sys


class FilterEffects:
    """Class containing static methods for applying various effects."""

    @staticmethod
    def apply_cartoon_effect(frame):
        """Apply a cartoon effect."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        return cv2.bitwise_and(color, color, mask=edges)

    @staticmethod
    def apply_rainbow_effect(frame, t):
        """Apply a moving rainbow effect."""
        height, width = frame.shape[:2]
        rainbow = frame.copy()
        for i in range(width):
            hue = (i + t) % 180
            rainbow[:, i] = cv2.addWeighted(frame[:, i], 0.7,
                                            np.full_like(frame[:, i], [hue, 255, 255]), 0.3, 0)
        return cv2.cvtColor(cv2.cvtColor(rainbow, cv2.COLOR_BGR2HSV), cv2.COLOR_HSV2BGR)


def load_overlay_image(image_path):
    """Load an overlay image with transparency."""
    overlay_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if overlay_img is None:
        print(f"Error: Could not load overlay image from {image_path}")
        sys.exit(1)

    # Add alpha channel if missing
    if overlay_img.shape[2] == 3:
        overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2BGRA)
    return overlay_img


def overlay_transparent(background, overlay, x, y, overlay_size=None):
    """Overlay a transparent PNG onto the background image."""
    if overlay.shape[2] != 4:
        print("Warning: Overlay image doesn't have an alpha channel")
        return background

    if overlay_size:
        overlay = cv2.resize(overlay, overlay_size)

    h, w = overlay.shape[:2]

    # Adjust overlay position if it goes out of bounds
    if y < 0:
        overlay = overlay[-y:, :]
        h += y
        y = 0
    if x < 0:
        overlay = overlay[:, -x:]
        w += x
        x = 0
    if y + h > background.shape[0]:
        overlay = overlay[:background.shape[0] - y, :]
        h = background.shape[0] - y
    if x + w > background.shape[1]:
        overlay = overlay[:, :background.shape[1] - x]
        w = background.shape[1] - x

    if h <= 0 or w <= 0:
        return background

    roi = background[y:y + h, x:x + w]
    alpha = overlay[:, :, 3] / 255.0
    alpha = np.dstack((alpha, alpha, alpha))

    # Blend overlay with background
    foreground = overlay[:, :, :3] * alpha
    background_part = roi * (1 - alpha)
    background[y:y + h, x:x + w] = foreground + background_part
    return background


def apply_edge_detection(frame):
    """Apply Canny edge detection to the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(frame, 0.7, edges_colored, 0.3, 0)


def main():
    # Load face and eye cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    if face_cascade.empty():
        print("Error: Could not load face cascade classifier")
        sys.exit(1)

    # Load overlay images
    overlays = {
        '1': load_overlay_image("filters/sunglasses.png"),
        '2': load_overlay_image("filters/cool_mask.png"),
        '3': load_overlay_image("filters/hat.png")
    }

    current_overlay = overlays['1']
    current_effect = None
    rainbow_t = 0
    show_edges = False

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        sys.exit(1)

    print("""
AR Face Filter started. Controls:
1-3: Switch overlays
c: Cartoon effect
r: Rainbow effect
e: Toggle edge detection
n: Normal mode
q: Quit
    """)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        processed_frame = frame.copy()

        # Apply current effect
        if current_effect == 'cartoon':
            processed_frame = FilterEffects.apply_cartoon_effect(processed_frame)
        elif current_effect == 'rainbow':
            processed_frame = FilterEffects.apply_rainbow_effect(processed_frame, rainbow_t)
            rainbow_t = (rainbow_t + 5) % 180

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Apply edge detection if enabled
        if show_edges:
            processed_frame = apply_edge_detection(processed_frame)

        # Process each detected face
        for (x, y, w, h) in faces:
            if current_overlay is overlays['1']:  # Sunglasses
                overlay_width, overlay_height = w, int(h / 3)
                overlay_x, overlay_y = x, y + int(h / 4)
            elif current_overlay is overlays['2']:  # Mask
                overlay_width, overlay_height = w, int(h / 2)
                overlay_x, overlay_y = x, y + int(h / 2)
            elif current_overlay is overlays['3']:  # Hat
                overlay_width, overlay_height = int(w * 1.2), int(h / 2)
                overlay_x, overlay_y = x - int(w * 0.1), y - int(h / 2)

            # Overlay the current filter
            processed_frame = overlay_transparent(
                processed_frame,
                current_overlay,
                overlay_x,
                overlay_y,
                (overlay_width, overlay_height)
            )

        # Display the result
        cv2.imshow('AR Face Filter', processed_frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('1'), ord('2'), ord('3')]:
            current_overlay = overlays.get(chr(key), current_overlay)
        elif key == ord('e'):
            show_edges = not show_edges
        elif key == ord('c'):
            current_effect = 'cartoon'
        elif key == ord('r'):
            current_effect = 'rainbow'
        elif key == ord('n'):
            current_effect = None

    cap.release()
    cv2.destroyAllWindows()
    print("AR Face Filter stopped.")


if __name__ == "__main__":
    main()