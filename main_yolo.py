import cv2
import argparse
import numpy as np

class yolo():
    def __init__(self, config):
        print('Net use', config['netname'])
        self.confThreshold = config['confThreshold']
        self.nmsThreshold = config['nmsThreshold']
        self.inpWidth = config['inpWidth']
        self.inpHeight = config['inpHeight']
        with open(config['classesFile'], 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        self.colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(self.classes))]
        self.net = cv2.dnn.readNet(config['modelConfiguration'], config['modelWeights'])

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=4)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        return frame

    # Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)

    def detect(self, srcimg):
        blob = cv2.dnn.blobFromImage(srcimg, 1/255.0, (self.inpWidth, self.inpHeight), [0, 0, 0], swapRB=True, crop=False)
        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        self.postprocess(srcimg, outs)
        return srcimg

Net_config = [{'confThreshold':0.5, 'nmsThreshold':0.4, 'inpWidth':416, 'inpHeight':416, 'classesFile':'coco.names', 'modelConfiguration':'yolov3/yolov3.cfg', 'modelWeights':'yolov3/yolov3.weights', 'netname':'yolov3'},
              {'confThreshold':0.5, 'nmsThreshold':0.4, 'inpWidth':608, 'inpHeight':608, 'classesFile':'coco.names', 'modelConfiguration':'yolov4/yolov4.cfg', 'modelWeights':'yolov4/yolov4.weights', 'netname':'yolov4'},
              {'confThreshold':0.5, 'nmsThreshold':0.4, 'inpWidth':320, 'inpHeight':320, 'classesFile':'coco.names', 'modelConfiguration':'yolo-fastest/yolo-fastest-xl.cfg', 'modelWeights':'yolo-fastest/yolo-fastest-xl.weights', 'netname':'yolo-fastest'},
              {'confThreshold':0.5, 'nmsThreshold':0.4, 'inpWidth':320, 'inpHeight':320, 'classesFile':'coco.names', 'modelConfiguration':'yolobile/csdarknet53s-panet-spp.cfg', 'modelWeights':'yolobile/yolobile.weights', 'netname':'yolobile'}]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='bus.jpg', help='image path')
    parser.add_argument('--net_type', default=0, type=int, choices=[0, 1, 2, 3])
    args = parser.parse_args()

    yolonet = yolo(Net_config[args.net_type])
    srcimg = cv2.imread(args.imgpath)
    srcimg = yolonet.detect(srcimg)

    winName = 'Deep learning object detection in OpenCV'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
