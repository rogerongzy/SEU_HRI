from aip import AipFace
import base64
import cv2



# pip3 install baidu-aip
# def connect():


# convert to BASE64
def image_to_base64(img_dir):
    image_np = cv2.imread(img_dir)
    image = cv2.imencode('.jpg',image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code






APP_ID = '25077207'
API_KEY = '4XyzQekdDRPQ3WMZvHr4bHRq'
SECRET_KEY = 'GS3Rr9gj9VkbKKMY6yAMqBFrUIhDndd4'
client = AipFace(APP_ID, API_KEY, SECRET_KEY)

# main
# 取决于image_type参数，传入BASE64字符串或URL字符串或FACE_TOKEN字符串
image = image_to_base64('dev.jpg')
imageType = "BASE64"

# optional params (referring)
options = {}
options["face_field"] = "age,gender"
options["max_face_num"] = 3


print(client.detect(image, imageType, options)['result']['face_list'][0]['age'])
print(client.detect(image, imageType, options)['result']['face_list'][0]['gender']['type'])
