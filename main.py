import codecs
import json

import numpy as np
from recognition import FaceRecognition
# pip install cmake dlib==19.22

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
Open_Dump = True

if __name__ == '__main__':
    fr = FaceRecognition(openfromFile=Open_Dump)
    if (not Open_Dump):
        f =  open("dump.json", "w")
        dump = json.dumps(
            {
            "known_face_encodings": fr.known_face_encodings,
            "known_face_names": fr.known_face_names,
            }, 
                        cls=NumpyEncoder) 
        
        f.writelines(dump)
        f.close()
    fr.run_recognition()





"""
TO restore NDarr from json:

json_load = json.loads(json_dump)
a_restored = np.asarray(json_load["a"])
print(a_restored)
print(a_restored.shape)

"""