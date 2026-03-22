import sys
import cv2
import numpy as np
import pygame
import pymunk
import random
from pymunk import Vec2d

# ========================== AYARLAR ==========================
SCREEN_W, SCREEN_H = 800, 600
FPS = 60
DT = 1.0 / FPS
SEGMENT_RADIUS = 10
INITIAL_SEGMENTS = 8
SEGMENT_DISTANCE = 15
SNAKE_SPEED = 500.0

# ========================== ELMA TEMİZLEME FONKSİYONU ==========================
def temizle_resim(path_in, path_out="elma_clean.png"):
    img = cv2.imread(path_in, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Elma görseli bulunamadı!")
        return None
    lower = np.array([200, 200, 200])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(img[:, :, :3], lower, upper)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img[mask > 0] = (0, 0, 0, 0)
    cv2.imwrite(path_out, img)
    return path_out

# ========================== COLOR TRACKER ==========================
class ColorTracker:
    def __init__(self, cam_index=0):
        self.cap = cv2.VideoCapture(cam_index)
        self.lower_color = np.array([100, 150, 50])
        self.upper_color = np.array([140, 255, 255])
        self.latest_pos = None

    def get_position(self):
        ret, frame = self.cap.read()
        if not ret: return None
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 500:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    px = int(M["m10"] / M["m00"])
                    py = int(M["m01"] / M["m00"])
                    h, w = frame.shape[:2]
                    gx = (px / w) * SCREEN_W
                    gy = (py / h) * SCREEN_H
                    self.latest_pos = (gx, gy)
                    return self.latest_pos
        return self.latest_pos

    def stop(self):
        self.cap.release()

# ========================== SNAKE CLASS ==========================
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

# ========================== GAME CLASS ==========================
class Game:
    def __init__(self):
        pygame.init()
        info = pygame.display.Info()
        global SCREEN_W, SCREEN_H
        SCREEN_W, SCREEN_H = info.current_w, info.current_h
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))

        self.arkaplan = pygame.image.load("arkaplan.png").convert_alpha()
        self.arkaplan = pygame.transform.smoothscale(self.arkaplan, (SCREEN_W, SCREEN_H))

        # Elma görselini yükle
        temiz_png = temizle_resim("elma.png")
        self.food_img = pygame.image.load(temiz_png).convert_alpha()
        self.food_img = pygame.transform.smoothscale(self.food_img, (28, 28))

        self.clock = pygame.time.Clock()
        self.space = pymunk.Space()
        self.snake = Snake(self.space, (SCREEN_W // 2, SCREEN_H // 2))
        self.food_pos = Vec2d(random.randint(50, SCREEN_W - 50), random.randint(50, SCREEN_H - 50))
        self.tracker = ColorTracker()

        self.running = True
        self.game_over = False
        self.score = 0

        self.font = pygame.font.SysFont("Arial", 48)
        self.small_font = pygame.font.SysFont("Arial", 28)

        self.create_walls()

    def create_walls(self):
        thickness = 20
        walls = [
            ((SCREEN_W/2, thickness/2), (-SCREEN_W/2, 0), (SCREEN_W/2, 0)),
            ((SCREEN_W/2, SCREEN_H - thickness/2), (-SCREEN_W/2, 0), (SCREEN_W/2, 0)),
            ((thickness/2, SCREEN_H/2), (0, -SCREEN_H/2), (0, SCREEN_H/2)),
            ((SCREEN_W - thickness/2, SCREEN_H/2), (0, -SCREEN_H/2), (0, SCREEN_H/2))
        ]
        for pos, start, end in walls:
            wall = pymunk.Body(body_type=pymunk.Body.STATIC)
            wall.position = pos
            shape = pymunk.Segment(wall, start, end, thickness)
            self.space.add(wall, shape)

    def reset_game(self):
        self.snake = Snake(self.space, (SCREEN_W // 2, SCREEN_H // 2))
        self.food_pos = Vec2d(random.randint(50, SCREEN_W - 50), random.randint(50, SCREEN_H - 50))
        self.score = 0
        self.game_over = False

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    if event.key == pygame.K_r and self.game_over:
                        self.reset_game()

            if self.game_over:
                self.screen.blit(self.arkaplan, (0, 0))
                go_text = self.font.render("GAME OVER", True, (255, 0, 0))
                score_text = self.small_font.render(f"Skor: {self.score}", True, (255, 255, 255))
                restart_text = self.small_font.render("Tekrar başlamak için R tuşuna basın", True, (200, 200, 200))

                self.screen.blit(go_text, (SCREEN_W//2 - 160, SCREEN_H//2 - 80))
                self.screen.blit(score_text, (SCREEN_W//2 - 80, SCREEN_H//2))
                self.screen.blit(restart_text, (SCREEN_W//2 - 200, SCREEN_H//2 + 50))
                pygame.display.flip()
                self.clock.tick(60)
                continue

            # Normal oyun
            self.screen.blit(self.arkaplan, (0, 0))

            pos = self.tracker.get_position()
            if pos:
                self.snake.target = Vec2d(*pos)
                pygame.draw.circle(self.screen, (0, 100, 255), (int(pos[0]), int(pos[1])), 5)

            self.snake.update(DT)
            self.space.step(DT)

            # Yemek yeme
            if (self.snake.segments[0].position - self.food_pos).length < 25:
                self.snake.add_segment(self.snake.segments[-1].position)
                self.food_pos = Vec2d(random.randint(50, SCREEN_W - 50), random.randint(50, SCREEN_H - 50))
                self.score += 1

            # Duvara çarpma
            head = self.snake.segments[0]
            if (head.position.x < 30 or head.position.x > SCREEN_W - 30 or
                head.position.y < 30 or head.position.y > SCREEN_H - 30):
                self.game_over = True

            # Çizimler
            self.screen.blit(self.food_img, (int(self.food_pos.x) - 14, int(self.food_pos.y) - 14))
            self.snake.draw(self.screen)

            score_text = self.small_font.render(f"Skor: {self.score}", True, (255, 255, 255))
            self.screen.blit(score_text, (20, 20))

            pygame.display.flip()
            self.clock.tick(FPS)

        self.tracker.stop()
        pygame.quit()


if __name__ == "__main__":
    Game().run()
