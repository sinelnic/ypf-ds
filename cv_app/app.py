import os
import json

from flask import Flask, jsonify
from flask_restful import reqparse, abort, Api, Resource

from imageai.Detection import ObjectDetection

execution_path = os.getcwd()
upload_dir = os.path.join(execution_path, "images")
models_dir = os.path.join(execution_path, "models")

if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

app = Flask(__name__)
api = Api(app)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

class Filer(Resource):    
    def get(self):
        """Endpoint to list files on the server."""
        files = []
        for filename in os.listdir(upload_dir):
            path = os.path.join(upload_dir , filename)
            if os.path.isfile(path):
                files.append(filename)
        return jsonify(files)

class DetectObjects(Resource):
    def __init__(self):
        self.detector = ObjectDetection()
        self.detector.setModelTypeAsRetinaNet()
        self.detector.setModelPath( os.path.join(models_dir , "resnet50_coco_best_v2.0.1.h5"))
        self.detector.loadModel()

    def get(self, image_name):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']

        #Detect objects in image
        detections = self.detector.detectObjectsFromImage(input_image=os.path.join(upload_dir , image_name), 
                                               output_image_path=os.path.join(upload_dir , 'image-out.jpg'))

        # create JSON object
        output = []
        for eachObject in detections:
            incl_keys = ['name','percentage_probability']
            element = { k: eachObject[k] for k in set(incl_keys) & set(eachObject.keys()) }
            output.append( element )
        return jsonify(output)

# Setup the Api resource routing here
# Route the URL to the resource

api.add_resource(Filer, '/')
api.add_resource(DetectObjects, '/detect/<string:image_name>')


if __name__ == '__main__':
    app.run(debug=True)