import numpy as np
import tflite_runtime.interpreter as tflite
from io import BytesIO
from urllib import request
from PIL import Image


def download_image(url: str):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)  # , Resampling.NEAREST)
    return img


def preprocess_image(img):
    x = np.array(img, dtype="float32")

    X = np.array([x])

    X *= 1.0 / 255

    # X /= 127.5
    # X -= 1.
    return X


def from_url(url):
    test_img = download_image(url)
    test_img = prepare_image(test_img, target_size=target_size)

    X = preprocess_image(test_img)
    return X


tflite_model_path = "./model_2024_hairstyle_v2.tflite"
# tflite_model_path = "./models/model_2024_hairstyle.tflite"

interpreter = tflite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
_ = int(interpreter.get_input_details()[0]["shape_signature"][1])
target_size = (_, _)


# https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg


def predict(url):
    classes = ["placeholder"]

    X = from_url(url)

    interpreter.set_tensor(input_index, X)

    interpreter.invoke()

    pred_y = interpreter.get_tensor(output_index)
    float_preds = pred_y.tolist()

    return dict(zip(classes, float_preds[0]))


def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)

    return result
