from __future__ import print_function
import requests
import json
import cv2

#url = 'https://face-recog-api-6dacyht5fa-uc.a.run.app/'
url='https://face-recog-s36y6zfosq-uc.a.run.app/'


# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('me.jpg')
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)
print(img_encoded)
# send http request with image and receive response
response = requests.post(url, data=img_encoded.tostring(), headers=headers)
# decode response
print(json.loads(response.text))