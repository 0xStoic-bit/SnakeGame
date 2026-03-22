# open Cv kütüphanesi ile mevi renk cisimleri takip eden basit yılan oyunu

import sys
import cv2
import numpy as np
import pygame
import pymunk
import random
from pymunk import Vec2d
import sys


# -------------------------
# Ayarlar
# -------------------------
SCREEN_W, SCREEN_H = 800, 600
FPS = 60
DT = 1.0 / FPS
SEGMENT_RADIUS = 10
INITIAL_SEGMENTS = 8
SEGMENT_DISTANCE = 15
SNAKE_SPEED = 500.0



#ekran arkaplan bloğu




# -------------------------
# ColorTracker
# -------------------------
class ColorTracker:
    def __init__(self, cam_index=0):
        self.cap = cv2.VideoCapture(cam_index)
        # MAVİ RENK ARALIĞI (HSV formatında)

        self.lower_color = np.array([100, 150, 50])
        self.upper_color = np.array([140, 255, 255])
        self.latest_pos = None

    def get_position(self):
        ret, frame = self.cap.read()
        if not ret: return None

        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Belirlenen rengi maskele
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # En büyük renk kütlesini bul
        contours, _ = cv2.find_external_contours(mask) if hasattr(cv2, 'find_external_contours') else cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 500:  # Çok küçük noktaları görmezden gel
                M = cv2.moments(c)
                if M["m00"] != 0:
                    px = int(M["m10"] / M["m00"])
                    py = int(M["m01"] / M["m00"])

                    # Kamera çözünürlüğünü oyun ekranına oranla
                    h, w = frame.shape[:2]
                    gx = (px / w) * SCREEN_W
                    gy = (py / h) * SCREEN_H
                    self.latest_pos = (gx, gy)
                    return self.latest_pos
        return self.latest_pos

    def stop(self):
        self.cap.release()


# -------------------------
# Yılan ve Oyun Mantığı
# -------------------------
class Snake:
    def __init__(self, space, start_pos):
        self.space = space
        self.segments = []
        self.target = Vec2d(*start_pos)
        for i in range(INITIAL_SEGMENTS):
            self.add_segment(Vec2d(*start_pos))

    def add_segment(self, pos):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = pos
        shape = pymunk.Circle(body, SEGMENT_RADIUS)
        shape.collision_type = 2
        self.space.add(body, shape)
        self.segments.append(body)

    def update(self, dt):
        head = self.segments[0]
        direction = self.target - head.position
        if direction.length > 10:
            head.velocity = direction.normalized() * SNAKE_SPEED
        else:
            head.velocity = 0, 0

        for i in range(1, len(self.segments)):
            prev, curr = self.segments[i - 1].position, self.segments[i].position
            if (prev - curr).length > SEGMENT_DISTANCE:
                self.segments[i].position = prev - (prev - curr).normalized() * SEGMENT_DISTANCE

    def draw(self, screen):
        for i, b in enumerate(self.segments):
            color = (0, 255, 150) if i == 0 else (0, 180, 80)
            pygame.draw.circle(screen, color, (int(b.position.x), int(b.position.y)), SEGMENT_RADIUS)


class Game:
    def __init__(self):
        pygame.init()
        info=pygame.display.Info()
        SCREEN_W,SCREEN_H = info.current_w, info.current_h
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))

        #temel properties
        self.arkaplan = pygame.image.load("arkaplan.png").convert_alpha()
        self.arkaplan = pygame.transform.smoothscale(self.arkaplan, (SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()
        self.space = pymunk.Space()
        self.snake = Snake(self.space, (400, 300))
        self.food_pos = Vec2d(random.randint(50, 750), random.randint(50, 550))
        self.tracker = ColorTracker()
        self.running = True

        #font kısmı
        self.font = pygame.font.SysFont("Arial", 20)
        #score ataması
        self.score = 0

    def run(self):
        while self.running:

            #self.screen.fill((15, 15, 15))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                #Esc ye basarsa çıksın
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False


            #Ekranı çiz
            self.screen.blit(self.arkaplan, (0, 0))

            # Renk takibi ile hedefi güncelle
            pos = self.tracker.get_position()
            if pos:
                self.snake.target = Vec2d(*pos)
                pygame.draw.circle(self.screen, (0, 100, 255), (int(pos[0]), int(pos[1])), 5)

            # Fizik
            self.snake.update(DT)
            self.space.step(DT)

            # Yemek yeme kontrolü
            if (self.snake.segments[0].position - self.food_pos).length < 20:
                self.snake.add_segment(self.snake.segments[-1].position)
                self.food_pos = Vec2d(random.randint(50, 750), random.randint(50, 550))
                self.score += 1


            # Çizim
            pygame.draw.circle(self.screen, (255, 50, 50), (int(self.food_pos.x), int(self.food_pos.y)), 10)
            self.snake.draw(self.screen)

            score_text = self.font.render(f"score : {self.score}", True, (255, 255, 255))
            self.screen.blit(score_text, (10, 10))

            pygame.display.flip()
            self.clock.tick(FPS)

        self.tracker.stop()
        pygame.quit()


if __name__ == "__main__":
    Game().run()
