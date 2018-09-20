from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "models/resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "images/image.jpg"),
                                    output_image_path=os.path.join(execution_path , "images/image-out.jpg"))

output = []
for eachObject in detections:
    incl_keys=['name','percentage_probability']
    element = {k: eachObject[k] for k in set(incl_keys) & set(eachObject.keys()) }
    output.append( element )
print(output)