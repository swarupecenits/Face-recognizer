import streamlit as st
import face_recognition
import cv2
import os
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.5  # Adjust the resizing factor for better performance

    def load_encoding_images(self, images_path):
        for dirpath, dirnames, filenames in os.walk(images_path):
            for file in filenames:
                if file.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    image_path = os.path.join(dirpath, file)
                    image = face_recognition.load_image_file(image_path)
                    encoding = face_recognition.face_encodings(image)
                    if len(encoding) > 0:
                        self.known_face_encodings.append(encoding[0])
                        self.known_face_names.append(os.path.basename(dirpath))
                    else:
                        print(f"No encoding found for {file}")
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names

def main():
    st.title("Face Recognition App")
    st.write("Upload images to encode faces and recognize them in real-time from webcam feed.")

    images_path = st.text_input("Enter the path to the images directory:", "images/")
    if st.button("Load Encodings", key="load_encodings"):
        sfr = SimpleFacerec()
        sfr.load_encoding_images(images_path)
        st.success("Encodings loaded successfully")

    run_webcam = st.button("Start Webcam", key="start_webcam")

    if run_webcam:
        sfr = SimpleFacerec()
        sfr.load_encoding_images(images_path)

        cap = cv2.VideoCapture(0)
        stop = st.button("Stop Webcam", key="stop_webcam")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            face_locations, face_names = sfr.detect_known_faces(frame)
            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame, channels="RGB")

            if stop:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
