import numpy as np


class MobileUser:
    def __init__(self, x=0, y=50, speed=10, direction=0):
        self.x = x
        self.y = y
        self.speed = speed
        self.direction = direction

    def move(self, time_step=1):
        direction_rad = np.radians(self.direction)

        self.x += self.speed * np.cos(direction_rad) * time_step
        self.y += self.speed * np.sin(direction_rad) * time_step

        return self.x, self.y