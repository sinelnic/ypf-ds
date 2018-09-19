import os
import json

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource

from imageai.Detection import ObjectDetection

execution_path = os.getcwd()

app = Flask(__name__)
api = Api(app)

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "models/resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

class DetectObjects(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']

        #Detect objects in image
        detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "images/image.jpg"),
                                    output_image_path=os.path.join(execution_path , "images/image-out.jpg"))

        # create JSON object
        output = json.dumps(detections)

        for eachObject in detections:
            print(eachObject["name"] , " : " , eachObject["percentage_probability"] )

        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(DetectObjects, '/')


if __name__ == '__main__':
    app.run(debug=True)