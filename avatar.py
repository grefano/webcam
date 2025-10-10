import cv2
import mediapipe as mp
import numpy as np

class PointAvatar:
    def __init__(self, id, label="") -> None:
        self.id = id
        self.label = str(id) if label=="" else label
        self.camX = -1
        self.camY = -1
        self.avatarX = -1
        self.avatarY = -1

    def setCamPosition(self, x, y):
        self.camX = x
        self.camY = y

    def toPoint(self):
        return (self.avatarX, self.avatarY)

    

class PropertyAvatar:
    def __init__(self, label, fn, ids) -> None:
        self.value = 0
        self.label = label
        self.fn = fn
        self.ids = ids
    def evaluate(self, points):
        self.value = self.fn(points) 


class Avatar:
    def __init__(self, detector) -> None:
        self.points = {}
        self.properties = {}
        self.imgCam: cv2.UMat | None = None 
        self.imgAvatar = None
        self.detector = detector
        self.faces_landmarks = []

    def updateImg(self, imgCam, imgAvatar, faces_landmarks):
        self.imgCam = imgCam
        self.imgAvatar = imgAvatar
        self.faces_landmarks = faces_landmarks
    def createPoint(self, id, label=""):
        self.points[str(id)] = PointAvatar(id, label)
    def getPoint(self, id) -> PointAvatar:
        return self.points[str(id)]   
    
    def updatePointsCam(self):
        # print(f"pointS {self.points}")
        for point in self.points.values():
            # print(f"point {point}")
            (camX, camY) = self.getPointCamPosition(point.id)
            
            point.camX = camX
            point.camY = camY
    def updatePointsAvatar(self):
        for point in self.points.values():
            (avatarX, avatarY) = self.getPointAvatarPosition(point.id)
            lerp = .9
            point.avatarX = int(point.avatarX + (avatarX-point.avatarX) * lerp)
            point.avatarY = int(point.avatarY + (avatarY-point.avatarY) * lerp)
            # cv2.circle(self.imgAvatar, (point.avatarX, point.avatarY), 0, (0,0,255), 5) #type: ignore

    def createProperty(self, label, fn, ids):
        self.properties[label] = PropertyAvatar(label, fn, ids)

    def updateProperties(self):
        # print(f"propertieS {self.properties}")
        for p in self.properties.values():
            points = []
            for id in p.ids:
                points.append(self.getPoint(id))
            p.evaluate(points)
            # print(f"property {p.value}")

    def getProperty(self, label):
        return self.properties[label].value
    
    def getPointCamPosition(self, id: int):
        
        return self.detector.getPointPositionById(id, self.faces_landmarks[0], self.imgCam) #type: ignore

    def getPointAvatarPosition(self, id):
        pointOrigin = self.getPoint(152)
        (x, y) = self.getPointCamPosition(id)
        faceDir = self.getProperty("faceDir")
        faceH = self.getProperty("faceH")

        newxtrans = x-pointOrigin.camX
        newytrans = y-pointOrigin.camY
        
        
        cosdir = np.cos(-faceDir)
        sindir = np.sin(-faceDir)

        newxrot = newxtrans * cosdir - newytrans * sindir
        newyrot = newxtrans * sindir + newytrans * cosdir

        newxnorm = newxrot / faceH #div / (faceHeight/H)#
        newynorm = newyrot / faceH

        cv2.circle(self.imgCam, (int(pointOrigin.camX+newxnorm*240), int(pointOrigin.camY+newynorm*240)), 2, (100, 0, 0), 2) #type: ignore

        size = 1000
        return int(size/2 + newxnorm*size), int(size - newynorm*size)
