import cv2
import mediapipe as mp
import numpy as np
import time

import threading
import queue
from avatar import Avatar 
from facemeshdetector import FaceMeshDetector
from draw import *
class Point:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y




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



# def ConfigOnMouse(event, x, y, flags, params):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         (id, x, y, dist) = detector.getPointByScreenPosition(x, y)
#         # global pointMouseClicked
#         # pointMouseClicked = (id, x, y, dist)
#         windowConfig.point_clicked = (id, x, y, dist)
#         print(f"mouse clicou {x} {y}")

# def PreviewOnMouse(event, x, y, flags, params):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         windowPreview.point_clicked = detector.getPointByScreenPosition(x, y)

#     if event == cv2.EVENT_RBUTTONDOWN:
#         (id, x, y, dist) = detector.getPointByScreenPosition(x, y)


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
    skipFrames = 2
    resultsLast = None
    while True:
        _, imgWebcam = webcam.read()
        imgConfig = imgWebcam.copy()
        imgPreview = np.full((1000, 1000, 3), (0, 255, 0), dtype=np.uint8)
        # frameSmall = cv2.resize(imgWebcam, (320, 240))
        if frameCount % skipFrames == 0:
            frameSmall = cv2.resize(imgWebcam, (160, 120))
            results = detector.process(frameSmall)
            if results and results.multi_face_landmarks:
                resultsLast = results
        if resultsLast:  # Só atualiza se tiver resultado novo
            # avatar.faces_landmarks = results.multi_face_landmarks

            avatar.updateImg(imgWebcam, imgPreview, resultsLast.multi_face_landmarks)
        frameCount+=1
    


    # while True:
        
    #     _, imgWebcam = webcam.read()
    #     imgConfig = imgWebcam.copy()
    #     imgPreview = np.full((1000, 1000, 3), (0, 255, 0), dtype=np.uint8)
    #     if True:
            
    #         H,W,C = imgWebcam.shape


    #         # frameSmall = cv2.resize(imgWebcam, (320, 240))
    #         frameSmall = cv2.resize(imgWebcam, (320, 240))

            

    #         # faces = detector.findFaceMesh(frameSmall)
            
    #         faces_landmarks = detector.faceMesh.process(frameSmall).multi_face_landmarks
    #         # imgPreview = cv2.resize(imgConfig, (1000, 1000))
    #         avatar.updateImg(imgWebcam, imgPreview, faces_landmarks)
        
        frameCount+=1
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        
        cv2.putText(imgConfig, f"fps: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 50), 2)


        (_, x, y, _) = windowConfig.point_clicked
        cv2.circle(imgConfig, (x,y), 2, (255, 100, 100), 4)


        if avatar.faces_landmarks:

            avatar.updatePointsCam()
            avatar.updateProperties()
            avatar.updatePointsAvatar()
            


            headRecoil = 100
            headTopRecoil = 100+headRecoil
            chinRecoil = 100+headRecoil
            cv2.fillPoly(imgPreview, [np.array([(0, headRecoil), (0, 999), (999-chinRecoil, 999), (999-headRecoil, 700+headRecoil), (999-headRecoil, 150+headRecoil), (999-headTopRecoil, headRecoil)])], (0, 255, 255))

            #desenhando pontos
            
            drawOlho(173, 130, 159, 23, avatar, imgPreview)
            drawOlho(463, 359, 253, 386, avatar, imgPreview)


            drawBoca(291, 61, 0, 17, avatar, imgPreview)
            
            # detector.mpDraw.draw_landmarks(
            #     image=imgConfig,
            #     landmark_list=avatar.faces_landmarks[0],
            #     connections=detector.mpFaceMesh.FACEMESH_TESSELATION,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=detector.mpDrawingStyles.get_default_face_mesh_tesselation_style()
            # )

            cv2.imshow("preview", imgPreview)
            cv2.putText(imgConfig, f"faceDir {avatar.getProperty('faceDir')/np.pi*180}", (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
            # cv2.putText(imgConfig, f"face {faceWidth+faceHeight}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        cv2.imshow("config", imgConfig)
        
        # cv2.setMouseCallback("config", ConfigOnMouse)
        # cv2.setMouseCallback("preview", PreviewOnMouse)
        if cv2.waitKey(5) == 27:
            break
        
    detector.stop()
    webcam.release()
    cv2.destroyAllWindows()

