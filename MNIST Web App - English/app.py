from flask import Flask, render_template, request
from keras.models import model_from_json
import re, os, base64, cv2, numpy as np
from PIL import Image

from load import *

app = Flask(__name__)

"""
global model, graph
# initialize these variables
model, graph = init()
"""
# Loading the model
PATH = "model/"
json_file = open(PATH + "model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
# load weights into new model
model.load_weights(PATH + "model.h5")
print("Loaded model from disk")


@app.route('/')
def index_view():
    # return jsonify(str('all fine! shree'))
    return render_template('index.html')

def convertImage(imgData):
	imgstr = re.search(b'base64,(.*)',imgData).group(1)
	with open('output.png','wb') as output:
	    output.write(base64.b64decode(imgstr))

@app.route('/predict/',methods=['POST'])
def predict():
    imgData = request.get_data()
    convertImage(imgData)
    img = Image.open("output.png").convert("L")
    img = img.resize((28, 28))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1, 28, 28, 1)
    out = model.predict(im2arr)
    return str(np.argmax(out))
    
    
if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
      "please wait until server has fully started"))
    app.run(debug=False, port=8000, threaded=False)
