import queue
import threading
import cv2
import mediapipe as mp
class FaceMeshDetector:
    def __init__(self, refine_landmarks=False, max_num_faces=1, min_detection_confidence=0.1, min_tracking_confidence=0.1) -> None:
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.multi_face_landmarks = []

        self.mpFaceMesh = mp.solutions.face_mesh #type: ignore
        self.mpDraw = mp.solutions.drawing_utils #type: ignore
        self.mpDrawingStyles = mp.solutions.drawing_styles #type: ignore
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)
        # self.faceMesh = self.mpFaceMesh.FaceMesh(refine_landmarks=self.refine_landmarks, max_num_faces=self.max_num_faces, min_detection_confidence=self.min_detection_confidence, min_tracking_confidence=self.min_tracking_confidence)
        # Filas para threading
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.multi_face_landmarks = []
        
        # Iniciar thread
        self.thread = threading.Thread(target=self._process_thread, daemon=True)
        self.running = True
        self.thread.start()

    def _process_thread(self, refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        faceMesh = self.mpFaceMesh.FaceMesh(
            refine_landmarks=refine_landmarks,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        while self.running:
            try:
                # Bloqueia até ter frame (não fica em loop vazio)
                frame = self.frame_queue.get(timeout=0.1)
                if frame is None:
                    break
                
                results = faceMesh.process(frame)
                
                # Limpa fila de resultados se estiver cheia
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.result_queue.put(results)
            except queue.Empty:
                continue
            
        faceMesh.close()

    def process(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.frame_queue.full():
            self.frame_queue.get()  # Remove frame antigo
        self.frame_queue.put(rgb_frame)
        
        # Retorna último resultado disponível
        if not self.result_queue.empty():
            results = self.result_queue.get()
            self.multi_face_landmarks = results.multi_face_landmarks if results.multi_face_landmarks else []
            return results
        
        return None  # Ainda processando
    
    def stop(self):
        self.running = False
        self.frame_queue.put(None)
        self.thread.join()
    
    def getPointPositionById(self, id, landmarkList, img):
        l = landmarkList.landmark[id]
        H, W, C = img.shape
        return int(l.x*W), int(l.y*H)
    
    def getPointPositionByLandmark(self, landmark, img):
        H, W, C = img.shape
        return int(landmark.x*W), int(landmark.y*H)
    
    # def getPointByScreenPosition(self, x, y):
    #     global faces_landmarks
    #     nearestId, nearestX, nearestY, nearestDist = (-1, -1, -1, -1)
    #     for id, landmark in enumerate(faces_landmarks[0].landmark):
    #         H,W,_ = imgWebcam.shape
    #         lx, ly = int(landmark.x*W), int(landmark.y*H)
    #         dist = np.sqrt(np.power(x-lx, 2) + np.power(y-ly, 2))
    #         if dist < nearestDist or nearestDist == -1:
    #             nearestId = id
    #             nearestX = lx
    #             nearestY = ly
    #             nearestDist = dist
    #     return (nearestId, int(nearestX), int(nearestY), int(nearestDist))