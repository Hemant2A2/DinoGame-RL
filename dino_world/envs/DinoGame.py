import gym
from gym import spaces
import numpy as np
import pygame
import os
import random

# Initialize Pygame
pygame.init()

# dimentions of the screen
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100

# load the images of the dinosaur, obstacles and the background
RUNNING = [pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))]
JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))
DUCKING = [pygame.image.load(os.path.join("Assets/Dino", "DinoDuck1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoDuck2.png"))]

SMALL_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))]
LARGE_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png"))]

BIRD = [pygame.image.load(os.path.join("Assets/Bird", "Bird1.png")),
        pygame.image.load(os.path.join("Assets/Bird", "Bird2.png"))]

CLOUD = pygame.image.load(os.path.join("Assets/Other", "Cloud.png"))
BG = pygame.image.load(os.path.join("Assets/Other", "Track.png"))


class Dinosaur:
    X_POS = 80
    Y_POS = 310
    Y_POS_DUCK = 340
    JUMP_VEL = 8.5

    def __init__(self):
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_vel = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        # top left x,y coordinates of the rectangle surrounding the dino
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    def update(self, action):
        if self.dino_duck:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 10:
            self.step_index = 0

        if action == 0 and not self.dino_jump:  # Jump
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif action == 1 and not self.dino_jump:  # Run
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False
        elif action == 2 and not self.dino_jump:  # Duck
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False

    def duck(self):
        self.image = self.duck_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel * 4
            self.jump_vel -= 0.8
        if self.jump_vel < - self.JUMP_VEL:
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))

    def get_position(self):
        return self.dino_rect.x, self.dino_rect.y, self.dino_duck, self.dino_jump


class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD
        self.width = self.image.get_width()

    def update(self):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))


class Obstacle:
    def __init__(self, image, type):
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            obstacles.pop()

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)

    def get_position(self):
        return self.rect.x, self.rect.y


class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 325


class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 300


class Bird(Obstacle):
    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)
        self.rect.y = 250
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 9:
            self.index = 0
        SCREEN.blit(self.image[self.index//5], self.rect)
        self.index += 1


class DinoGameEnv(gym.Env):
    def __init__(self):
        global game_speed, obstacles
        # list to store the obstacles when near to the dino
        obstacles = []
        # game speed increases as the game progresses
        game_speed = 20
        super(DinoGameEnv, self).__init__()
         # [Jump,Run, Duck] are the elemnts of the action space
        self.action_space = spaces.Discrete(3)
        # observation space is the top left x,y coordinates of the rectangle surrounding the dino,the state of the dino (ducking or jumping) 
        # and the top left x,y coordinates of the rectangle surrounding the obstacle
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0, 0]), high=np.array([SCREEN_WIDTH, SCREEN_HEIGHT, 1, 1, SCREEN_WIDTH, SCREEN_HEIGHT]), dtype=np.float32)
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Dino Run")
        self.clock = pygame.time.Clock()
        # initialize the dinosaur and cloud objects
        self.player = Dinosaur()
        self.cloud = Cloud()
        # initialize the Track position
        self.x_pos_bg = 0
        self.y_pos_bg = 380
        # the longer the game runs, the higher the points
        self.points = 0
        self.font = pygame.font.Font('freesansbold.ttf', 20)
        # the game ends when the dino collides with an obstacle
        self.done = False
        self.reward = 0
        self.info = {}

    def step(self, action):
        global game_speed, obstacles

        self.player.update(action)
        if len(obstacles) == 0:
            # randomly generate obstacles in the environment
            if random.randint(0, 2) == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS))
            elif random.randint(0, 2) == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS))
            elif random.randint(0, 2) == 2:
                obstacles.append(Bird(BIRD))

        for obstacle in obstacles:
            obstacle.update()
            if self.player.dino_rect.colliderect(obstacle.rect):
                pygame.time.delay(2000)
                # reduction in reward if the dino collides with an obstacle
                # the second term is to ensure that dino that run longer get a higher reward
                self.reward -= (5 + 100 / self.points)
                self.done = True
            else:
                # additonal reward for avoiding the obstacles
                self.reward += 1

        self.cloud.update()
        self.points += 1
        if self.points % 100 == 0:
            game_speed += 1

        # increase in reward as the game progresses
        self.reward += self.points / 100.0

        return self._get_observation() , self.reward, self.done, self.info

    def reset(self):
        global game_speed, obstacles
        obstacles = []
        game_speed = 20
        self.player = Dinosaur()
        self.cloud = Cloud()
        self.x_pos_bg = 0
        self.y_pos_bg = 380
        self.points = 0
        self.done = False
        self.reward = 0
        return self._get_observation()

    def render(self):
        global game_speed, obstacles
        self.screen.fill((255, 255, 255))
        self.player.draw(self.screen)

        for obstacle in obstacles:
            obstacle.draw(self.screen)

        self.cloud.draw(self.screen)
        self._draw_background()
        self._draw_score()
        self.clock.tick(30)
        pygame.display.update()

    def _draw_background(self):
        global game_speed
        image_width = BG.get_width()
        self.screen.blit(BG, (self.x_pos_bg, self.y_pos_bg))
        self.screen.blit(BG, (image_width + self.x_pos_bg, self.y_pos_bg))
        if self.x_pos_bg <= -image_width:
            self.screen.blit(BG, (image_width + self.x_pos_bg, self.y_pos_bg))
            self.x_pos_bg = 0
        self.x_pos_bg -= game_speed

    def _draw_score(self):
        text = self.font.render("Points: " + str(self.points), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (1000, 40)
        self.screen.blit(text, textRect)

    def _get_observation(self):
        global obstacles
        dino_x, dino_y, dino_duck, dino_jump = self.player.get_position()
        if obstacles:
            obstacle_x, obstacle_y = obstacles[0].get_position()
        else:
            obstacle_x, obstacle_y = SCREEN_WIDTH, 0  # No obstacle present
        return np.array([dino_x, dino_y, dino_duck, dino_jump, obstacle_x, obstacle_y], dtype=np.float32)

    def close(self):
        pygame.quit()
