import os
import json

from flask import Flask, jsonify
from flask_restful import reqparse, abort, Api, Resource

from imageai.Detection import ObjectDetection

execution_path = os.getcwd()

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
        dir = os.path.join(execution_path, "images")
        for filename in os.listdir(dir):
            path = os.path.join(dir , filename)
            if os.path.isfile(path):
                files.append(filename)
        return jsonify(files)

class DetectObjects(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']

        #Detect objects in image
        detector = ObjectDetection()
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath( os.path.join(execution_path , "models/resnet50_coco_best_v2.0.1.h5"))
        detector.loadModel()

        detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "images/image.jpg"), 
                                               output_image_path=os.path.join(execution_path , "images/image-out.jpg"))

        # create JSON object
        return jsonify(detections)


# Setup the Api resource routing here
# Route the URL to the resource

api.add_resource(Filer, '/')
api.add_resource(DetectObjects, '/detect')


if __name__ == '__main__':
    app.run(debug=True)