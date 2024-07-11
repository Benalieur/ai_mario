import random
import math

from classes.Collider import Collider


class LeftRightWalkTrait:
    def __init__(self, entity, level):
        self.direction = random.choice([-1, 1])
        self.entity = entity
        self.collDetection = Collider(self.entity, level)
        self.speed = 1
        self.entity.vel.x = self.speed * self.direction

    def update(self):
        if self.entity.vel.x == 0:
            self.direction *= -1
        self.entity.vel.x = self.speed * self.direction
        self.moveEntity()

    def moveEntity(self):
        # Convertir les coordonnées en indices de grille
        grid_y = self.entity.rect.y // 32
        grid_x = self.entity.rect.x // 32

        # Logs pour le débogage

        # Vérifier les limites avant de déplacer l'entité
        if grid_y < 0 or grid_y >= len(self.collDetection.level):
            self.entity.alive = None
            return

        self.entity.rect.y += self.entity.vel.y
        self.collDetection.checkY()

        if grid_x < 0 or grid_x >= len(self.collDetection.level[0]):
            self.entity.alive = None
            return

        self.entity.rect.x += self.entity.vel.x
        self.collDetection.checkX()