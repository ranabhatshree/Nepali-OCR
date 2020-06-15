from keras.models import model_from_json
import tensorflow as tf
import keras

def init():
    PATH = "model/"
    json_file = open(PATH  + 'model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model

    loaded_model.load_weights(PATH + "model.h5")
    print("Loaded Model from disk")

    #compile and evaluate loaded model
    loaded_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

    graph = tf.compat.v1.get_default_graph()

    return loaded_model, graph
