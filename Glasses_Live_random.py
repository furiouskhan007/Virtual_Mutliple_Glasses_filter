import cv2
import dlib
import numpy as np

num = 1

# Load the face detector and landmark detector models
face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the virtual sunglasses image
virtual_sunglasses = cv2.imread('Glasses/glass{}.png'.format(num), cv2.IMREAD_UNCHANGED)

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    k = cv2.waitKey(100)
    if k == ord('s'):
        num = num + 1
        if num <= 29:
            virtual_sunglasses = cv2.imread('Glasses/glass{}.png'.format(num), cv2.IMREAD_UNCHANGED)

    # Capture frame from the webcam
    ret, frame = cap.read()

    # Detect the user's face in the frame
    face_rects = face_detector(frame)
    if len(face_rects) > 0:
        # Get the facial landmarks for the first face detected
        face_landmarks = landmark_detector(frame, face_rects[0])
        left_eye_pos = np.array([face_landmarks.part(36).x, face_landmarks.part(36).y])
        right_eye_pos = np.array([face_landmarks.part(45).x, face_landmarks.part(45).y])

        # Calculate the position and size of the sunglasses relative to the user's face
        glasses_width = np.linalg.norm(left_eye_pos - right_eye_pos) * 1.85
        glasses_height = glasses_width * virtual_sunglasses.shape[0] / virtual_sunglasses.shape[1]
        glasses_center = (left_eye_pos + right_eye_pos) / 2
        glasses_offset = np.array([-glasses_width // 2, -glasses_height // 2])

        # Resize the virtual sunglasses image to match the calculated size
        virtual_sunglasses_resized = cv2.resize(virtual_sunglasses, (int(glasses_width), int(glasses_height)))

        # Overlay the virtual sunglasses on the frame
        overlay_img = frame.copy()
        rows, cols, channels = virtual_sunglasses_resized.shape
        roi = overlay_img[
              int(glasses_center[1] + glasses_offset[1]):int(glasses_center[1] + glasses_offset[1] + rows),
              int(glasses_center[0] + glasses_offset[0]):int(glasses_center[0] + glasses_offset[0] + cols)]
        alpha = virtual_sunglasses_resized[:, :, 3] / 255.0
        overlay_img[
        int(glasses_center[1] + glasses_offset[1]):int(glasses_center[1] + glasses_offset[1] + rows),
        int(glasses_center[0] + glasses_offset[0]):int(glasses_center[0] + glasses_offset[0] + cols)] = \
            (alpha[:, :, np.newaxis] * virtual_sunglasses_resized[:, :, :3] + (
                    1 - alpha[:, :, np.newaxis]) * roi).astype(np.uint8)

        # Display the frame with virtual sunglasses
        cv2.imshow('Virtual Sunglasses', overlay_img)
    else:
        # No face detected, display the original frame
        cv2.imshow('Virtual Sunglasses', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q') or num > 29:
        break

cap.release()
cv2.destroyAllWindows()
