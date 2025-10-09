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


if __name__ == '__main__':

    webcam = cv2.VideoCapture(0)
    windowConfig = WindowPoints("config", webcam)
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
    while True:
        
        
        _, imgWebcam = webcam.read()
        H,W,C = imgWebcam.shape


        frameSmall = cv2.resize(imgWebcam, (320, 240))

        

        # faces = detector.findFaceMesh(frameSmall)
        
        faces_landmarks = detector.faceMesh.process(frameSmall).multi_face_landmarks
        faces = []
        imgConfig = imgWebcam.copy()
        imgPreview = cv2.resize(imgConfig, (1000, 1000))
        avatar.updateImg(imgWebcam, imgPreview, faces_landmarks)
        
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

