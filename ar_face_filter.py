import cv2
import numpy as np
import os
import sys
from pathlib import Path

class FilterEffects:
    @staticmethod
    def apply_cool_effect(frame):
        """Apply a cool blue tint effect"""
        blue_tint = frame.copy()
        blue_tint[:, :, 0] = np.clip(blue_tint[:, :, 0] * 1.5, 0, 255)  # Enhance blue
        return blue_tint
    
    @staticmethod
    def apply_warm_effect(frame):
        """Apply a warm orange tint effect"""
        warm_tint = frame.copy()
        warm_tint[:, :, 1] = np.clip(warm_tint[:, :, 1] * 1.2, 0, 255)  # Enhance green
        warm_tint[:, :, 2] = np.clip(warm_tint[:, :, 2] * 1.4, 0, 255)  # Enhance red
        return warm_tint
    
    @staticmethod
    def apply_cartoon_effect(frame):
        """Apply a cartoon effect"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon

    @staticmethod
    def apply_rainbow_effect(frame, t):
        """Apply a moving rainbow effect"""
        height, width = frame.shape[:2]
        rainbow = frame.copy()
        for i in range(width):
            hue = (i + t) % 180
            rainbow[:, i] = cv2.addWeighted(frame[:, i], 0.7, 
                                          np.full_like(frame[:, i], [hue, 255, 255]), 0.3, 0)
        return cv2.cvtColor(cv2.cvtColor(rainbow, cv2.COLOR_BGR2HSV), cv2.COLOR_HSV2BGR)

def load_overlay_image(image_path):
    """Load the overlay image with transparency"""
    try:
        # Load the image
        overlay_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if overlay_img is None:
            print(f"Error: Could not load overlay image from {image_path}")
            print("Using a placeholder overlay instead.")
            # Create a placeholder overlay (simple sunglasses shape)
            overlay_img = np.zeros((100, 200, 4), dtype=np.uint8)
            # Draw rectangle for frame
            cv2.rectangle(overlay_img, (0, 0), (199, 99), (0, 0, 0, 255), 5)
            # Draw lenses
            cv2.circle(overlay_img, (50, 50), 40, (0, 0, 0, 255), -1)
            cv2.circle(overlay_img, (150, 50), 40, (0, 0, 0, 255), -1)
            # Add transparency
            overlay_img[:, :, 3] = np.where(np.any(overlay_img[:, :, :3] > 0, axis=2), 255, 0)
        else:
            # Check if image has an alpha channel
            if overlay_img.shape[2] == 3:  # If it's a 3-channel image (no alpha)
                # Convert to 4 channels by adding an alpha channel
                rgba = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2BGRA)
                # Set fully opaque
                rgba[:, :, 3] = 255
                overlay_img = rgba
                
        return overlay_img
    except Exception as e:
        print(f"Error loading overlay image: {e}")
        sys.exit(1)

def overlay_transparent(background, overlay, x, y, overlay_size=None):
    """Overlay a transparent PNG onto the background image"""
    try:
        # Ensure overlay has 4 channels (BGRA)
        if overlay.shape[2] != 4:
            print("Warning: Overlay image doesn't have an alpha channel")
            return background
            
        # Resize overlay if size is provided
        if overlay_size is not None:
            overlay = cv2.resize(overlay, overlay_size)
        
        # Get dimensions
        h, w = overlay.shape[:2]
        
        # Ensure coordinates are within background image
        if y < 0:
            overlay = overlay[-y:, :]
            h += y
            y = 0
        if x < 0:
            overlay = overlay[:, -x:]
            w += x
            x = 0
            
        # Check if overlay goes beyond background boundaries
        if y + h > background.shape[0]:
            overlay = overlay[:background.shape[0] - y, :]
            h = background.shape[0] - y
        if x + w > background.shape[1]:
            overlay = overlay[:, :background.shape[1] - x]
            w = background.shape[1] - x
        
        # Extract region of interest
        if h <= 0 or w <= 0:
            return background
            
        roi = background[y:y+h, x:x+w]
        
        try:
            # Extract alpha channel and create masks
            alpha = overlay[:h, :w, 3] / 255.0
            alpha = np.dstack((alpha, alpha, alpha))
            
            # Apply alpha blending
            foreground = overlay[:h, :w, :3] * alpha
            background_part = roi * (1 - alpha)
            
            # Combine foreground and background
            result = background.copy()
            result[y:y+h, x:x+w] = foreground + background_part
            
            return result
        except Exception as e:
            print(f"Error in alpha blending: {e}")
            return background
            
    except Exception as e:
        print(f"Error in overlay_transparent: {e}")
        return background

def apply_edge_detection(frame):
    """Apply Canny edge detection to the frame"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Convert back to BGR for blending
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Blend original frame with edges
        result = cv2.addWeighted(frame, 0.7, edges_colored, 0.3, 0)
        
        return result
    except Exception as e:
        print(f"Error in edge detection: {e}")
        return frame

def main():
    # Load face cascade
    try:
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        if face_cascade.empty():
            print("Error: Could not load face cascade classifier")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading face cascade: {e}")
        sys.exit(1)
    
    # Load eye cascade for better positioning of glasses
    try:
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    except Exception as e:
        print(f"Warning: Could not load eye cascade: {e}")
        eye_cascade = None
    
    # Load different overlay images
    overlays = {
        '1': load_overlay_image("filters/sunglasses.png"),
        '2': load_overlay_image("filters/party_glasses.png"),
        '3': load_overlay_image("filters/cool_mask.png"),
        '4': load_overlay_image("filters/hat.png")
    }
    
    current_overlay = overlays['1']  # Default overlay
    current_effect = None  # No effect by default
    rainbow_t = 0  # Time variable for rainbow effect
    show_edges = False
    
    # Initialize webcam
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            sys.exit(1)
    except Exception as e:
        print(f"Error initializing webcam: {e}")
        sys.exit(1)
    
    print("""
AR Face Filter started. Controls:
1-4: Switch overlays
e: Toggle edge detection
c: Cartoon effect
w: Warm effect
b: Cool blue effect
r: Rainbow effect
n: Normal mode
q: Quit
    """)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Create a copy for effects
        processed_frame = frame.copy()
        
        # Apply current effect
        if current_effect == 'cartoon':
            processed_frame = FilterEffects.apply_cartoon_effect(processed_frame)
        elif current_effect == 'warm':
            processed_frame = FilterEffects.apply_warm_effect(processed_frame)
        elif current_effect == 'cool':
            processed_frame = FilterEffects.apply_cool_effect(processed_frame)
        elif current_effect == 'rainbow':
            processed_frame = FilterEffects.apply_rainbow_effect(processed_frame, rainbow_t)
            rainbow_t = (rainbow_t + 5) % 180
        
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Apply edge detection if enabled
        if show_edges:
            edge_result = apply_edge_detection(processed_frame)
            processed_frame = cv2.addWeighted(processed_frame, 0.7, edge_result, 0.3, 0)
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw rectangle around the face (for debugging)
            cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Determine overlay position and size based on the type of overlay
            if current_overlay is overlays['1'] or current_overlay is overlays['2']:  # Glasses
                overlay_width = w
                overlay_height = int(h / 3)
                overlay_x = x
                overlay_y = y + int(h / 4)
                
                # Use eye detection for better positioning
                if eye_cascade is not None:
                    roi_gray = gray[y:y+h, x:x+w]
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    if len(eyes) >= 2:
                        eyes = sorted(eyes, key=lambda e: e[0])
                        eye1_x, eye1_y, eye1_w, eye1_h = eyes[0]
                        eye2_x, eye2_y, eye2_w, eye2_h = eyes[1]
                        overlay_y = y + min(eye1_y, eye2_y)
                        overlay_width = int((eye2_x + eye2_w - eye1_x) * 1.5)
                        overlay_x = x + eye1_x - int(overlay_width * 0.1)
            
            elif current_overlay is overlays['3']:  # Mask
                overlay_width = w
                overlay_height = int(h / 2)
                overlay_x = x
                overlay_y = y + int(h / 2)
            
            elif current_overlay is overlays['4']:  # Hat
                overlay_width = int(w * 1.2)
                overlay_height = int(h / 2)
                overlay_x = x - int(w * 0.1)
                overlay_y = y - int(h / 2)
            
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
        elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
            current_overlay = overlays.get(chr(key), current_overlay)
        elif key == ord('e'):
            show_edges = not show_edges
        elif key == ord('c'):
            current_effect = 'cartoon'
        elif key == ord('w'):
            current_effect = 'warm'
        elif key == ord('b'):
            current_effect = 'cool'
        elif key == ord('r'):
            current_effect = 'rainbow'
        elif key == ord('n'):
            current_effect = None
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("AR Face Filter stopped.")

if __name__ == "__main__":
    main()