from __future__ import print_function
import requests
import json
import cv2
import numpy as np
import jsonpickle
import time
url='https://face-recog-s36y6zfosq-uc.a.run.app/'
video_capture = cv2.VideoCapture(0)
content_type = 'image/jpeg'
headers = {'content-type': content_type}

process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    
    if process_this_frame:
        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post(url, data=img_encoded.tostring(), headers=headers)
        print(response.text)
        result=json.loads(response.text)
        time.sleep(3)
        
    #     if len(result['loc'])>0:
    #         face_names=result['names']
    #         print(face_names)
    #         face_locations=result['loc'][0]['py/tuple']
    #         print(face_locations)
        

    #         process_this_frame = not process_this_frame


    #         # Display the results
    #         #for (top, right, bottom, left), name in zip(face_locations, face_names):
    #             # Scale back up face locations since the frame we detected in was scaled to 1/4 size
    #         top=face_locations[0]*4
    #         right=face_locations[1]*4
    #         bottom=face_locations[2]*4
    #         left=face_locations[3]*4
    #         name=face_names[0]
    #             # Draw a box around the face
    #         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    #             # Draw a label with a name below the face
    #         cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    #         font = cv2.FONT_HERSHEY_DUPLEX
    #         cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()