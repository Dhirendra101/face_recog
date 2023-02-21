
import json
from flask import Flask, render_template, Response, request, jsonify
import cv2
import face_recognition
import numpy as np
import jsonpickle
app=Flask(__name__)
#camera = cv2.VideoCapture(0)
# Load a sample picture and learn how to recognize it.
dhirendra_image = face_recognition.load_image_file("dhirendra/dhirendra1.jpg")
dhirendra_face_encoding = face_recognition.face_encodings(dhirendra_image)[0]

# Load a second sample picture and learn how to recognize it.
bradley_image = face_recognition.load_image_file("Bradley/bradley.jpg")
bradley_face_encoding = face_recognition.face_encodings(bradley_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    dhirendra_face_encoding,
    bradley_face_encoding
]
known_face_names = [
    "Dhiru",
    "Bradely"
]


@app.route('/', methods=['POST'])
def index():  
    
    nparr=np.fromstring(request.data,np.uint8)
    print("image np: ",nparr)
    frame=cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
           
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            face_names.append(name)

    res={'names':face_names, 'loc':face_locations}  
    print("res: ", res)     
    response_pickled = jsonpickle.encode(res)

    return Response(response=response_pickled, status=200, mimetype="application/json")
            

# @app.route('/')
# def index():
#     return render_template('index.html')

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080)