import cv2
import mediapipe as mp
import numpy as np
import time

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

    # def findFaceMesh(self, img):
    #     self.multi_face_landmarks = self.faceMesh.process(img).multi_face_landmakrs
    # def findFaceMesh(self, img):
    #     results = self.faceMesh.process(img)
    #     faces = []
    #     if results.multi_face_landmarks:
    #         for faceLandmarks in results.multi_face_landmarks:
    #             face = []

    #             for id, landmark in enumerate(faceLandmarks.landmark):
    #                 x, y = self.getPointPositionByLandmark(id, landmark, img)
    #                 face.append(Point(id, x, y))
    #             faces.append(face)
    #     return faces

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
        print(f"cu {faces_landmarks[0]}")
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

class teste:
    def __init__(self, landmarks) -> None:
        self.landmark = landmarks

if __name__ == '__main__':

    webcam = cv2.VideoCapture(0)
    windowConfig = WindowPoints("config", webcam)
    windowPreview = WindowPoints("preview", webcam)


    detector = FaceMeshDetector(True, 1, 0.5, 0.5)
    faces_landmarks = []
    previousTime = 0    

    idSee = 180
    while True:
        
        
        _, imgWebcam = webcam.read()
        H,W,C = imgWebcam.shape


        frameSmall = cv2.resize(imgWebcam, (320, 240))

        

        # faces = detector.findFaceMesh(frameSmall)
        
        faces_landmarks = detector.faceMesh.process(frameSmall).multi_face_landmarks
        faces = []

        imgConfig = imgWebcam.copy()
        # imgPreview = imgWebcam.copy()
        
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        
        # imgConfig = imgNormal.copy()
        cv2.putText(imgConfig, f"fps: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 50), 2)
        # detector.mpDraw.draw_landmarks(
        #     image=imgConfig,
        #     landmark_list=,
        #     connections=detector.mpFaceMesh.FACEMESH_TESSELATION,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=detector.mpDrawingStyles.get_default_face_mesh_tesselation_style()
        # )

        (_, x, y, _) = windowConfig.point_clicked
        cv2.circle(imgConfig, (x,y), 2, (255, 100, 100), 4)

        # (_, x, y, _) = windowPreview.point_clicked
        # cv2.circle(imgPreview, (x,y), 2, (255, 100, 100), 4)

        size = 1000
        # points = [284, 54, 211, 431]
        points = [104, 333, 430, 135]
        idQueixo = 152
        idBochecha = 192
        idTestaCima = 10
        idTestaBaixo = 104

        idOlhoLCima = 159
        idOlhoLBaixo = 23
        idOlhoLEsquerda = 130
        idOlhoLDireita = 173
        idOlhoRCima = 386
        idOlhoRBaixo = 253
        idOlhoRDireita = 359
        idOlhoREsquerda = 463

        idBocaL = 291#320#57
        idBocaR = 61#409#287
        idBocaU = 0
        idBocaD = 17


        # points = 

        if faces_landmarks:
            for faceLandmarks in faces_landmarks:
                # detector.mpDraw.draw_landmarks(
                #     image=imgPreview,
                #     landmark_list=faceLandmarks,
                #     connections=detector.mpFaceMesh.FACEMESH_TESSELATION,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=detector.mpDrawingStyles.get_default_face_mesh_tesselation_style()
                # )
                for id, landmark in enumerate(faceLandmarks.landmark):
                    x, y = detector.getPointPositionByLandmark(landmark, imgWebcam)
                    # cv2.circle(imgConfig, (x, y), 2, (0, 255, 0), 1)
                    color = (0, 0, 255) if id in points else (0, 255, 0)
                    cv2.putText(imgConfig, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, color, 1)
        
        if faces_landmarks != None:
            def getPointPos(id: int):
                return detector.getPointPositionById(id, faces_landmarks[0], imgWebcam)

            pointQueixo = getPointPos(idQueixo)
            pointTestaCima = getPointPos(idTestaCima)
            pointTestaBaixo = getPointPos(idTestaBaixo)
            pointBochecha = getPointPos(idBochecha)
            pointBochechaR = getPointPos(365)

            faceHeight = np.sqrt((pointTestaCima[1]-pointQueixo[1])**2 + (pointQueixo[0]-pointTestaCima[0])**2)#abs(pointQueixo[1]-pointTestaCima[1])
            faceWidth = np.sqrt((pointBochechaR[1]-pointBochecha[1])**2 + (pointBochecha[0]-pointBochechaR[0])**2)#abs(pointBochechaR[0]-pointBochecha[0])
            cv2.circle(imgConfig,pointQueixo,0,(200,200,0), 10)
            cv2.circle(imgConfig,pointTestaCima,0,(200,200,0), 10)
            cv2.circle(imgConfig,pointBochecha,0,(200,200,0), 10)
            cv2.circle(imgConfig,pointBochechaR,0,(200,200,0), 10)
            # faceDir = -np.arctan2(int(pointQueixo[1]-pointTestaCima[1]), int(pointQueixo[0]-pointTestaCima[0]))
            faceDir = np.arctan2(int(pointTestaCima[0]-pointQueixo[0]), int(pointQueixo[1]-pointTestaCima[1]))
            print(f"face w {faceWidth}")
            def getPointNewPos(id: int):
                x, y = getPointPos(id)
                # cv2.circle(imgConfig, (x, y), 2, (0, 0, 255), 2)
                newxtrans = x-pointQueixo[0]
                newytrans = y-pointQueixo[1]
                
                
                cosdir = np.cos(-faceDir)
                sindir = np.sin(-faceDir)

                newxrot = newxtrans * cosdir - newytrans * sindir
                newyrot = newxtrans * sindir + newytrans * cosdir
                

                div = H
                newxnorm = newxrot / (faceHeight) #div / (faceHeight/H)#
                newynorm = newyrot / (faceHeight)
                
                newx = newxnorm
                newy = newynorm

                cv2.circle(imgConfig, (int(pointQueixo[0]+newx*faceWidth), int(pointQueixo[1]+newy*faceHeight)), 2, (0, 0, 255), 2)
                cv2.circle(imgConfig, (int(pointQueixo[0]+newxnorm*240), int(pointQueixo[1]+newynorm*240)), 2, (0, 0, 255), 2)

                mult = size

                return int(size/2 + newx*5*faceWidth), int(size + newy*5*faceHeight)
                return int(size/2 + newxnorm*mult), int(size + newynorm*mult)

            idSee += .12
            # print(f"id see {idSee}")
            # cv2.circle(imgConfig, getPointPos(int(idSee)), 2, (0, 0, 255), 2)

            cv2.line(imgConfig, pointQueixo, (int(pointQueixo[0]+np.sin(faceDir)*faceHeight), int(pointQueixo[1]+np.cos(faceDir)*faceHeight)), (0,0,0), 5)
            # pointOlhoLCima = getPointPos(idOlhoLCima)
            # pointOlhoLBaixo = getPointPos(idOlhoLBaixo)
            # pointOlhoRCima = getPointPos(idOlhoRCima)
            # pointOlhoRBaixo = getPointPos(idOlhoRBaixo)

            pointOlhoLCima = getPointNewPos(idOlhoLCima)
            pointOlhoLBaixo = getPointNewPos(idOlhoLBaixo)
            pointOlhoLEsquerda = getPointNewPos(idOlhoLEsquerda)
            pointOlhoLDireita = getPointNewPos(idOlhoLDireita)

            pointOlhoRCima = getPointNewPos(idOlhoRCima)
            pointOlhoRBaixo = getPointNewPos(idOlhoRBaixo)
            pointOlhoREsquerda = getPointNewPos(idOlhoREsquerda)
            pointOlhoRDireita = getPointNewPos(idOlhoRDireita)


            olhoaddx, olhoaddy, olhosepx = (150, 0, 0) 

            pointOlhoLCima = tuple(x + y for x, y in zip(pointOlhoLCima, (olhoaddx-olhosepx, olhoaddy))) 
            pointOlhoLBaixo = tuple(x + y for x, y in zip(pointOlhoLBaixo, (olhoaddx-olhosepx, olhoaddy))) 
            pointOlhoLEsquerda = tuple(x + y for x, y in zip(pointOlhoLEsquerda, (olhoaddx-olhosepx, olhoaddy))) 
            pointOlhoLDireita = tuple(x + y for x, y in zip(pointOlhoLDireita, (olhoaddx-olhosepx, olhoaddy))) 

            pointOlhoRCima = tuple(x + y for x, y in zip(pointOlhoRCima, (olhoaddx+olhosepx, olhoaddy))) 
            pointOlhoRBaixo = tuple(x + y for x, y in zip(pointOlhoRBaixo, (olhoaddx+olhosepx, olhoaddy))) 
            pointOlhoREsquerda = tuple(x + y for x, y in zip(pointOlhoREsquerda, (olhoaddx+olhosepx, olhoaddy))) 
            pointOlhoRDireita = tuple(x + y for x, y in zip(pointOlhoRDireita, (olhoaddx+olhosepx, olhoaddy))) 

            # print(f"olho left cima {pointOlhoLCima[0]} {pointOlhoLCima[1]}")
            # print(f"olho right cima {pointOlhoRCima}")

            pointBocaL = getPointNewPos(idBocaL)
            pointBocaR = getPointNewPos(idBocaR)
            pointBocaD = getPointNewPos(idBocaD)
            pointBocaU = getPointNewPos(idBocaU)
            # pointBocaLeft = 
            bocaaddx, bocaddy = (150, 0)
            pointBocaL = tuple(x + y for x, y in zip(pointBocaL, (bocaaddx, bocaddy))) 
            pointBocaR = tuple(x + y for x, y in zip(pointBocaR, (bocaaddx, bocaddy))) 
            pointBocaD = tuple(x + y for x, y in zip(pointBocaD, (bocaaddx, bocaddy))) 
            pointBocaU = tuple(x + y for x, y in zip(pointBocaU, (bocaaddx, bocaddy))) 


            # print(f"facedir {faceDir}")
            newpointTestaBaixo = getPointNewPos(idTestaBaixo)
            newpointTestaCima = getPointNewPos(idTestaCima)
            newpointBochecha = getPointNewPos(idBochecha)
            def getY(y):
                return int((y-newpointTestaCima[1]))
            yTestaBaixo = getY(newpointTestaBaixo[1])
            yBochecha = getY(newpointBochecha[1])
            # yOlhoL = getY(pointOlhoLCima[1])
            # yOlhoR = getY(pointOlhoRCima[1])
            # yOlhoM = (yOlhoL+yOlhoR)/2
            # yOlhoL = yOlhoM + (yOlhoL-yOlhoM)*faceDir
            # yOlhoR = yOlhoM - (yOlhoR-yOlhoM)*faceDir
            

            # yBoca = getY((pointBocaD[1]+pointBocaU[1])/2)
            # print(f"{yTestaBaixo} {yBochecha}")


            # srcPoints = np.array([detector.getPointPositionById(i, faces_landmarks[0], imgWebcam) for i in points], dtype=np.float32)
            # dstPoints = np.array([
            #     [0, 0],
            #     [size-1, 0],
            #     [size-1, size-1],
            #     [0, size-1]
            # ])
            # imgPreview = imgWebcam.copy()
            imgPreview = cv2.resize(imgWebcam, (size, size))
            # M, mask = cv2.findHomography(srcPoints, dstPoints, method=0)
            # imgPreview = cv2.warpPerspective(imgWebcam, M, (size, size))
            cv2.fillPoly(imgPreview, [np.array([(0,0), (int(size*.9), 0), (size-1, int(yTestaBaixo)), (int(size-1), int(yBochecha)), (int(size*.8), size-1), (0, size-1)])], (0,235,255))
            cv2.putText(imgPreview, f"ytestabaixo {yTestaBaixo} ybochecha {yBochecha}", (20, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0))
            def drawCircle(pos: cv2.typing.Point, radius=10, color=(0,0,0)):
                cv2.circle(imgPreview, pos, 0, color, radius)
            
            # drawCircle(pointOlhoLCima)
            # drawCircle(pointOlhoLBaixo)
            olhoRadius = 2
            olhoLRadius = abs(pointOlhoLBaixo[1]-pointOlhoLCima[1])
            if olhoLRadius > 20:
                drawCircle((int((pointOlhoLCima[0]+pointOlhoLBaixo[0])/2), int((pointOlhoLCima[1]+pointOlhoLBaixo[1])/2)), olhoLRadius*olhoRadius)
            else:
                cv2.line(imgPreview, pointOlhoLEsquerda, pointOlhoLDireita, (0, 0, 0), 10)
            # drawCircle(pointOlhoRCima)
            # drawCircle(pointOlhoRBaixo)
            olhoRRadius = abs(pointOlhoRBaixo[1]-pointOlhoRCima[1])
            if olhoRRadius > 20:
                drawCircle((int((pointOlhoRCima[0]+pointOlhoRBaixo[0])/2), int((pointOlhoRCima[1]+pointOlhoRBaixo[1])/2)), olhoRRadius*olhoRadius)
            else:
                pass
                cv2.line(imgPreview, pointOlhoREsquerda, pointOlhoRDireita, (0, 0, 0), 10)
            # cv2.ellipse(imgPreview, (300, 300), (20, 0), 0, 20, 180, (0, 0, 255), 10)

            bocaPointMiddle = (int((pointBocaR[0]+pointBocaL[0])/2), int((pointBocaD[1]+pointBocaU[1])/2))
            bocaRadius = abs(pointBocaD[1]-pointBocaU[1])
            if bocaRadius > 130:
                if bocaRadius > 100:
                    drawCircle(bocaPointMiddle, bocaRadius)
                else:
                    cv2.line(imgPreview, pointBocaL, pointBocaR, (0,0,0), 20)

            # drawCircle(pointBocaL)
            # drawCircle(pointBocaR)
            # drawCircle(pointBocaD)
            # drawCircle(pointBocaU)

            # drawCircle(pointOlhoLBaixo)
            # drawCircle(pointOlhoRCima)
            # drawCircle(pointOlhoRBaixo)

            # cv2.line(imgPreview, (int(size*.8), 0), (size-1, int(yTestaBaixo)), (0,0,0), 3)
            # drawCircle(int(size*.8), 0)
            # drawCircle(int(size-1), int(yTestaBaixo))
            # drawCircle(int(size-1), int(yBochecha))
            # drawCircle(int(size*.7), size-1)

            # drawCircle(int(size/2), int(yOlhoL), 10)
            # drawCircle(int(size*.75), int(yOlhoR), 10)

            # drawCircle(int(size*.6), int(getY(pointBocaU[1])), 10)
            # drawCircle(int(size*.6), int(yBoca), 10)
            # drawCircle(int(size*.6), int(getY(pointBocaD[1])), 10)

            # cv2.circle(imgPreview, (int(size*.8), 0), 3, (0, 0, 255), 2)
            # cv2.circle(imgPreview, (int(size-1), int(yTestaBaixo)), 3, (0, 0, 255), 2)
            # cv2.circle(imgPreview, (int(size-1), int(yBochecha)), 3, (0, 0, 255), 2)
            # cv2.circle(imgPreview, (int(size*.7), size-1), 3, (0, 0, 255), 2)

            

            cv2.imshow("preview", imgPreview)
            cv2.putText(imgConfig, f"faceDir {faceDir/np.pi*180}", (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
            cv2.putText(imgConfig, f"face {faceWidth+faceHeight}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        cv2.imshow("config", imgConfig)
        
        # cv2.setMouseCallback("config", ConfigOnMouse)
        # cv2.setMouseCallback("preview", PreviewOnMouse)
        if cv2.waitKey(5) == 27:
            break
        

    webcam.release()
    cv2.destroyAllWindows()