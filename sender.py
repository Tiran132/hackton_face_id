import time
import cv2 as cv
import requests
import os
 
encode_param = [int(cv.IMWRITE_JPEG_QUALITY), 90]
test_url = "http://localhost:5001/upload"

def name2id(name: str):
    return name.split(".")[0].split("-")[0]

def sendData(data: cv.typing.MatLike, arr):
    file_name = f'{round(time.time()*1000)}.jpg'
    cv.imwrite(file_name, data)

    f = open(file_name, "rb")

    ids = ""
    for i in range(len(arr)):
        elem = arr[i]
        if (i > 0): ids+=","
        ids += name2id(elem)

    print(ids)

    test_response = requests.post(test_url + "?uids=" + ids, files = {"form_field_name": f})
    print(test_response)
    f.close()

    os.remove(file_name)





# expample:
# 
# sendData(frame, ["egor1-1.jpg"])