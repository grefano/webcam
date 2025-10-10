import pygame
import numpy as np
import cv2
from avatar import Avatar
import threading



class ScreenWaiting:
    def __init__(self, avatar: Avatar) -> None:
        self.avatar = avatar
        self.pos = 100, 100
        self.spd = 5*16/9, 5
        self.w = 1280
        self.h = 720
        self.surface = None
        self.rect = None

    def cv2_to_surface(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.rot90(frame_rgb)
        surface = pygame.surfarray.make_surface(frame_rgb)
        return surface
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_thread, daemon=True)
        self.thread.start()
    def get_avatar_frame(self):
        # frame = np.zeros((300, 300, 3), dtype=np.uint8)
        try:

            frame = cv2.resize(self.avatar.imgAvatar, (200, 200)) #type: ignore
            if self.spd[0] > 0:
                frame = cv2.flip(frame, 1)
            return frame
        except Exception:
            return None
    def _run_thread(self):
        pygame.init()
        screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("live começa já")

        clock = pygame.time.Clock()
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            avatar_frame = self.get_avatar_frame()
            if avatar_frame is not None:
                self.surface = self.cv2_to_surface(avatar_frame)
                self.rect = self.surface.get_rect(topleft=self.pos)
                self.pos = tuple(p + s for p, s in zip(self.pos, self.spd))
                

                if self.pos[0] <= 0 or self.pos[0] + self.rect.width >= self.w:
                    self.spd = (-self.spd[0], self.spd[1])
                if self.pos[1] <= 0 or self.pos[1] + self.rect.height >= self.h:
                    self.spd = (self.spd[0], -self.spd[1])

            screen.fill((0, 0, 0))
            if avatar_frame is not None and self.surface is not None and self.rect is not None:
                screen.blit(self.surface, self.rect)
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)


    

