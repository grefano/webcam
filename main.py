import cv2
import mediapipe as mp
import numpy as np
import time
from avatar import Avatar 
class Point:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y


class FaceMeshDetector:
    def __init__(self, refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) -> None:
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        # self.multi_face_landmarks = []
        self.mpFaceMesh = mp.solutions.face_mesh #type: ignore
        self.mpDraw = mp.solutions.drawing_utils #type: ignore
        self.mpDrawingStyles = mp.solutions.drawing_styles #type: ignore
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)
        self.faceMesh = self.mpFaceMesh.FaceMesh(refine_landmarks=self.refine_landmarks, max_num_faces=self.max_num_faces, min_detection_confidence=self.min_detection_confidence, min_tracking_confidence=self.min_tracking_confidence)

    def getPointPositionById(self, id, landmarkList, img):
        l = landmarkList.landmark[id]
        H, W, C = img.shape
        return int(l.x*W), int(l.y*H)
    
    def getPointPositionByLandmark(self, landmark, img):
        H, W, C = img.shape
        return int(landmark.x*W), int(landmark.y*H)
    
    def getPointByScreenPosition(self, x, y):
        global faces_landmarks
        nearestId, nearestX, nearestY, nearestDist = (-1, -1, -1, -1)
        for id, landmark in enumerate(faces_landmarks[0].landmark):
            H,W,_ = imgWebcam.shape
            lx, ly = int(landmark.x*W), int(landmark.y*H)
            dist = np.sqrt(np.power(x-lx, 2) + np.power(y-ly, 2))
            if dist < nearestDist or nearestDist == -1:
                nearestId = id
                nearestX = lx
                nearestY = ly
                nearestDist = dist
        return (nearestId, int(nearestX), int(nearestY), int(nearestDist))


class Window:
    def __init__(self, name, img) -> None:
        self.name = name
        self.img = img


class WindowPoints(Window):
    def __init__(self, name, img) -> None:
        super().__init__(name, img)
        self.point_clicked = (int(-1), int(-1), int(-1), int(-1))
        self.points_selected = []



# pointMouseClicked = (-1, 10, 10, 0)



def ConfigOnMouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        (id, x, y, dist) = detector.getPointByScreenPosition(x, y)
        # global pointMouseClicked
        # pointMouseClicked = (id, x, y, dist)
        windowConfig.point_clicked = (id, x, y, dist)
        print(f"mouse clicou {x} {y}")

def PreviewOnMouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        windowPreview.point_clicked = detector.getPointByScreenPosition(x, y)

    if event == cv2.EVENT_RBUTTONDOWN:
        (id, x, y, dist) = detector.getPointByScreenPosition(x, y)


if __name__ == '__main__':

    webcam = cv2.VideoCapture(0)
    windowConfig = WindowPoints("config", 'webcam')
    windowPreview = WindowPoints("preview", webcam)

    detector = FaceMeshDetector(True, 1, 0.5, 0.5)
    
    avatar = Avatar(detector)


    # criando pontos    
    avatar.createPoint(152, "queixo")
    avatar.createPoint(192, "bochechaL")
    avatar.createPoint(416, "bochechaR")
    avatar.createPoint(10, "testaCima")
    avatar.createPoint(104, "testaBaixo")

    avatar.createPoint(159, "olhoLU")
    avatar.createPoint(23, "olhoLD")
    avatar.createPoint(130, "olhoLR")
    avatar.createPoint(173, "olhoLL")

    avatar.createPoint(386, "olhoRU")
    avatar.createPoint(253, "olhoRD")
    avatar.createPoint(359, "olhoRR")
    avatar.createPoint(463, "olhoRL")

    avatar.createPoint(291, "bocaL")
    avatar.createPoint(61, "bocaR")
    avatar.createPoint(0, "bocaU")
    avatar.createPoint(17, "bocaD")

    # criando propriedades do rosto
    avatar.createProperty("faceW",
        lambda points: np.sqrt((points[1].camY-points[0].camY)**2 + (points[0].camX-points[1].camX)**2)
        , [192, 416])
    avatar.createProperty("faceH",
        lambda points: np.sqrt((points[1].camY-points[0].camY)**2 + (points[0].camX-points[1].camX)**2)
        , [152, 10])
    avatar.createProperty("faceDir",
        lambda points: np.arctan2(int(points[0].camX-points[1].camX), int(points[1].camY-points[0].camY))
        , [152, 10])
    
    
    faces_landmarks = []
    previousTime = 0    

    idSee = 180
    frameCount = 0
    while True:
        
        _, imgWebcam = webcam.read()
        imgConfig = imgWebcam.copy()
        imgPreview = np.full((1000, 1000, 3), (0, 255, 0), dtype=np.uint8)
        if True:
            
            H,W,C = imgWebcam.shape


            # frameSmall = cv2.resize(imgWebcam, (320, 240))
            frameSmall = cv2.resize(imgWebcam, (320, 240))

            

            # faces = detector.findFaceMesh(frameSmall)
            
            faces_landmarks = detector.faceMesh.process(frameSmall).multi_face_landmarks
            faces = []
            # imgPreview = cv2.resize(imgConfig, (1000, 1000))
            avatar.updateImg(imgWebcam, imgPreview, faces_landmarks)
        
        frameCount+=1
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        
        cv2.putText(imgConfig, f"fps: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 50), 2)


        (_, x, y, _) = windowConfig.point_clicked
        cv2.circle(imgConfig, (x,y), 2, (255, 100, 100), 4)


        if faces_landmarks != None:

            avatar.updatePointsCam()
            avatar.updateProperties()
            avatar.updatePointsAvatar()
            


            headRecoil = 100
            headTopRecoil = 100+headRecoil
            chinRecoil = 100+headRecoil
            cv2.fillPoly(imgPreview, [np.array([(0, headRecoil), (0, 999), (999-chinRecoil, 999), (999-headRecoil, 700+headRecoil), (999-headRecoil, 150+headRecoil), (999-headTopRecoil, headRecoil)])], (0, 255, 255))

            #desenhando pontos
            def getMiddlePoint(ids):
                points = []
                for id in ids:
                    points.append(avatar.getPoint(id).toPoint())
                sumX = 0
                sumY = 0
                for point in points:
                    sumX += point[0]
                    sumY += point[1]

                return (int(sumX/len(points)), int(sumY/len(points)))
            def getMiddlePointExt(xIds, yIds):
                xPoints = []
                yPoints = []
                for id in xIds:
                    xPoints.append(avatar.getPoint(id).toPoint())
                for id in yIds:
                    yPoints.append(avatar.getPoint(id).toPoint())
                sumX = 0
                sumY = 0
                for point in xPoints:
                    sumX += point[0]
                for point in yPoints:
                    sumY += point[1]

                return (int(sumX/len(xPoints)), int(sumY/len(yPoints)))
            def getPointDistance(id1, id2):
                point1 = avatar.getPoint(id1).toPoint()
                point2 = avatar.getPoint(id2).toPoint()
                return np.sqrt((point2[0]-point1[0])**2 + (point2[1]-point1[1])**2)


            def drawOlho(idLeft, idRight, idUp, idDown):
                addx, addy = 200, 100

                olhoRadiusMult = 2
                blinkLineThickness = 12
                blinkBound = .25

                olhoSize = abs(avatar.getPoint(idDown).avatarY - avatar.getPoint(idUp).avatarY)
                olhoRatio = getPointDistance(idDown, idUp) / getPointDistance(idRight, idLeft)
                if (olhoRatio) > blinkBound:
                    (x, y) = getMiddlePointExt([idRight, idLeft], [idUp, idDown])
                    cv2.circle(imgPreview, (x+addx, y+addy), 0, (0,0,0), olhoSize*olhoRadiusMult)
                else:
                    (xl, yl) = avatar.getPoint(idLeft).toPoint()
                    (xr, yr) = avatar.getPoint(idRight).toPoint()
                    cv2.line(imgPreview, (xl + addx, yl + addy), (xr + addx, yr + addy), (0,0,0), blinkLineThickness)  

            drawOlho(173, 130, 159, 23)
            drawOlho(463, 359, 253, 386)

            def drawBoca(idLeft, idRight, idUp, idDown):
                addx, addy = 100, 100
                bocaRadiusMult = .01
                speakLineThickness = 36
                speakBound = .55
                smileBound = 350
                bocaW = getPointDistance(idRight, idLeft)
                bocaSize = abs(avatar.getPoint(idDown).avatarY - avatar.getPoint(idUp).avatarY)
                bocaRatio = getPointDistance(idDown, idUp) / bocaW
                print(f"boca ratio {bocaRatio}")
                if bocaW > smileBound:
                    cv2.line(imgPreview, tuple(p+addx for p in avatar.getPoint(idLeft).toPoint()), tuple(p+addx for p in avatar.getPoint(idRight).toPoint()), (0,0,0), speakLineThickness*3) 
                elif (bocaRatio) > speakBound:
                    cv2.circle(imgPreview, tuple(p+addx for p in getMiddlePointExt([idRight, idLeft], [idUp, idDown])), 0, (0,0,0), int((bocaSize**1.9)*bocaRadiusMult))
                else:
                    cv2.line(imgPreview, tuple(p+addx for p in avatar.getPoint(idLeft).toPoint()), tuple(p+addx for p in avatar.getPoint(idRight).toPoint()), (0,0,0), speakLineThickness) 
             
            drawBoca(291, 61, 0, 17)
            
            detector.mpDraw.draw_landmarks(
                image=imgConfig,
                landmark_list=faces_landmarks[0],
                connections=detector.mpFaceMesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=detector.mpDrawingStyles.get_default_face_mesh_tesselation_style()
            )

            cv2.imshow("preview", imgPreview)
            cv2.putText(imgConfig, f"faceDir {avatar.getProperty('faceDir')/np.pi*180}", (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
            # cv2.putText(imgConfig, f"face {faceWidth+faceHeight}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        cv2.imshow("config", imgConfig)
        
        # cv2.setMouseCallback("config", ConfigOnMouse)
        # cv2.setMouseCallback("preview", PreviewOnMouse)
        if cv2.waitKey(5) == 27:
            break
        

    webcam.release()
    cv2.destroyAllWindows()

