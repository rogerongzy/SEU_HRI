import face_recognition
import cv2
import base64
from aip import AipFace

# pip3 install dlib face_recognition (both latest version)
# pip3 install baidu-aip

def face_locate(img_dir):
    img = face_recognition.load_image_file(img_dir)
    # loc = face_recognition.face_locations(image)
    loc = face_recognition.face_locations(img, model="cnn") # using CNN for acute position
    return loc # top-left, down-right as (y,x), can be more than 1

def face_drawrec(img, loc):
    for loc_sp in loc:
        cv2.rectangle(img, (loc_sp[1], loc_sp[0]), (loc_sp[3], loc_sp[2]), (0, 255, 0), 2) # upside-down index 
    return img

def face_crop(img, loc): # named as real name 
    for idx, loc_sp in enumerate(loc):
        img_crop = img[loc_sp[0]:loc_sp[2], loc_sp[3]:loc_sp[1]]
        cv2.imwrite('crop_' + str(idx) + '.jpg', img_crop) ##### save using real name

def face_compare(img_dev_dir, img_input_dir):
    img_dev = face_recognition.load_image_file(img_dev_dir)
    img_input = face_recognition.load_image_file(img_input_dir)
    code_dev = face_recognition.face_encodings(img_dev)[0] # template cropped from the original pictures, len = 1
    code_input = face_recognition.face_encodings(img_input) # pictur for testing, len >= 1
    
    reco = []
    for code_input_mono in code_input:
        reco.append(face_recognition.compare_faces([code_dev], code_input_mono))
    
    loc_input = face_locate('test.jpg')
    return reco, loc_input

###########################

def image_to_base64(img_dir):
    image_np = cv2.imread(img_dir)
    image = cv2.imencode('.png',image_np)[1] # jpg or png
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code

def face_field_baiduAI(img_dir):
    APP_ID = '24094734'
    API_KEY = 'c1iTFhGzsaxag4sUSfza0A4A'
    SECRET_KEY = 'IGyZvfM7Hj2nGKxVEiKDwYt566DMnmp6'
    client = AipFace(APP_ID, API_KEY, SECRET_KEY)

    image = image_to_base64(img_dir)
    imageType = "BASE64"
    options = {}
    options["face_field"] = "age,gender" # with ,
    options["max_face_num"] = 3
    print("=========")
    print(client.detect(image, imageType, options)['result']['face_list'][0]['age'])
    print(client.detect(image, imageType, options)['result']['face_list'][0]['gender']['type'])

###########################

def test_demo():
    img = cv2.imread('dev.jpg')
    loc = face_locate('dev.jpg') ##### need to save first?
    face_crop(img, loc)
    
    reco_res, loc_res = face_compare('crop_0.jpg', 'test.jpg') ##### replace 'crop_0.jpg' with realname

    img_test = cv2.imread('test.jpg')
    img_result = face_drawrec(img_test, loc_res)

    print(reco_res)
    print(loc_res)

    cv2.imshow("result", img_test)
    cv2.waitKey(0)


# def landmark_ana():
#     image = face_recognition.load_image_file("dev.jpg")
#     face_landmarks_list = face_recognition.face_landmarks(image)
#     # face_landmarks_list is now an array with the locations of each facial feature in each face.
#     # face_landmarks_list[0]['left_eye'] would be the location and outline of the first person's left eye.
#     # chin, left_eyebrow, right_eyebrow, nose_bridge, nose_tip, left_eye, right_eye, top_lip, bottom_lip
#     print(face_landmarks_list[0]['left_eye'])

#     # method 1: curve fitting
#     # method 2: deep learning - diy / using BaiduAPI
    

if __name__ == '__main__':
    img = cv2.imread("multi.jpg")
    img = face_drawrec(img, face_locate("multi.jpg"))
    cv2.imshow('result', img)
    cv2.waitKey(0)
