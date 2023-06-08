import cv2


def highlightFace(net, frame, conf_threshold=0.5):
    frameOpencvDnn = frame.copy()

    frameHeight = frameOpencvDnn.shape[0]
    frameWIdth = frameOpencvDnn.shape[1]

    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)

    detections = net.forward()

    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence >= conf_threshold:
            x1 = int(detections[0,0,i,3]*frameWIdth)
            y1 = int(detections[0,0,i,4]*frameHeight)

            x2 = int(detections[0,0,i,5]*frameWIdth)
            y2 = int(detections[0,0,i,6]*frameHeight)

            faceBoxes.append([x1, y1, x2, y2])

            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 0)
    
    return frameOpencvDnn, faceBoxes

def findFaces(img_path=None):

    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    faceNet = cv2.dnn.readNet(faceModel, faceProto)

    if img_path:
        img = cv2.imread(img_path)
        resultImg, faceBoxes = highlightFace(faceNet, img)
        while cv2.waitKey(1) < 0:
            cv2.imshow("Face recognition", resultImg)

    else:
        video = cv2.VideoCapture(0)

        while cv2.waitKey(1) < 0:
            hasFrame, frame = video.read()

            if not hasFrame:
                cv2.waitKey
                break

            resultImg, faceBoxes = highlightFace(faceNet, frame)
    
            cv2.imshow("Face recognition", resultImg)

if __name__ == '__main__':
    findFaces('istockphoto-858269070-170667a.jpg')