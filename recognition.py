import json
import face_recognition
import os, sys
import cv2
import numpy as np
import math
import sender


# Helper
def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return round(linear_val * 100, 2)
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return round(value, 2)
    


id2names = [
    "Batomunkuev Pavel Maksimovich","Bystrushkin Evgenij Mihajlovich","Gavrilenko Mariya","Elyoskin Egor Evgenevich","Ershov Grigorij Arkadevich","Ivanenko Dmitrij Dmitrievich","Ivanov Aleksandr Aleksandrovich","Karataev Vecheslav","Kernoz Igor Sergeevich","Kozlova Anastasiya Genadevna","Korovkin Nikita","Kuznecova Polina Ivanovna","Kupriyanova Kristina Sergeevna","Ledengskaya Milana","Litvinenko Anastasiya Sergeevna","Matyushina Tatyana Sergeevna","Mitin Kirill Ivanovich","Myasnikov Lev Sergeevich","Nazarova YUliya Andreevna","Parhomenko Bogdan Borisovich","Pileckij Mihail Andreevich","Pochebyt Valeriya Evgenevna","Sidorov Vladislav Vladimirovich","Fomin Ilya Sergeevich","CHubarova Ekaterina","SHumakov Sergej","YAblonskij YAn Vitalevich","YArlykova YUliya","SHelupanov Aleksandr Aleksandrovich","undefied","Orlova Olga Aleksandrovna","Sulaev Evgenij Viktorovich","Efremov Aleksej YUrevich","Kolesov Ilya Borisovich"
]


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self, openfromFile=False):
        if (openfromFile):
            self.read_dump()
        else:
            self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            try:
                face_image = face_recognition.load_image_file(f"faces/{image}")
                face_encoding = face_recognition.face_encodings(face_image)[0]

                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(image)
            except: pass
        print(len(self.known_face_names))

    def read_dump (self):
        f =  open("dump.json", "r")
        lines = f.read()
        f.close()
        json_load = json.loads(lines)
        self.known_face_encodings = np.asarray(json_load["known_face_encodings"])
        self.known_face_names = np.asarray(json_load["known_face_names"])

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        while True:
            ret, frame = video_capture.read()

            frame_copy = frame.copy()


            # Only process every other frame of video to save time
            if self.process_current_frame:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

                # Find all the faces and face encodings in the current frame of video
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                tmp_names = [] 
                for face_encoding in self.face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = '???'

                    # Calculate the shortest distance to face
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])
                        if confidence > 80:
                            tmp_names.append(name)

                    if (name != "Unknown"):
                        self.face_names.append(f'{id2names[int(name.split("-")[0])-1]} ({confidence}%)')
                        print(f'{id2names[int(name.split("-")[0])-1]} - {confidence}')

            self.process_current_frame = not self.process_current_frame

            # Display the results
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                # Create the frame with the name
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            
            sender.sendData(frame_copy, tmp_names)
            # print(tmp_names)

            # Display the resulting image
            cv2.imshow('Face Recognition', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
