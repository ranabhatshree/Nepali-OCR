from flask import Flask, render_template, request
from keras.models import model_from_json
import re, os, base64, cv2, numpy as np
from PIL import Image

app = Flask(__name__)

IMG_WIDTH, IMG_HEIGHT = (32, 32)
nepali = ['0', '१', '२', '३', '४', '५', '६', '७', '८', '९']


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
    img = cv2.imread("output.png")
    img = np.resize(img, (IMG_WIDTH, IMG_HEIGHT, 3))
    img = img.reshape(1, IMG_WIDTH, IMG_HEIGHT, 3)
    out = model.predict(img)
    pred_val = nepali[np.argmax(out)]
    return str(pred_val)
    
    
if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
      "please wait until server has fully started"))
    app.run(debug=False, port=8000, threaded=False)
