import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

from datetime import datetime as dt
import json
import pygame
import cv2
import logging

from classes.Sprites import Sprites
from classes.Tile import Tile
from entities.Coin import Coin
from entities.CoinBrick import CoinBrick
from entities.Goomba import Goomba
from entities.Mushroom import RedMushroom
from entities.Koopa import Koopa
from entities.CoinBox import CoinBox
from entities.RandomBox import RandomBox

from sentiment_analysis.sentiment_analyzer import SentimentAnalyzer

class Level:
    def __init__(self, screen, sound, dashboard):
        self.sprites = Sprites()
        self.dashboard = dashboard
        self.sound = sound
        self.screen = screen
        self.level = None
        self.levelLength = 0
        self.entityList = []
        self.start_time = dt.now()
        self.frame = 0
        self.initial_length = 0
        self.max_pos_cam = 0
        self.n_extention = 0
        self.sentiment_analyzer = SentimentAnalyzer()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    def loadLevel(self, levelname):
        with open(f"./levels/{levelname}.json") as jsonData:
            data = json.load(jsonData)
            self.loadLayers(data)
            self.loadObjects(data)
            self.loadEntities(data)
            self.levelLength = data["length"]
        self.levelname = levelname
        self.initial_length = len(self.level[0])
        with open(f"./levels/{levelname}.json", "r") as jsonData:
            self.json_data = json.load(jsonData)

    def loadEntities(self, data):
        try:
            [self.addCoinBox(x, y) for x, y in data["level"]["entities"]["CoinBox"]]
            [self.addGoomba(x, y) for x, y in data["level"]["entities"]["Goomba"]]
            [self.addKoopa(x, y) for x, y in data["level"]["entities"]["Koopa"]]
            [self.addCoin(x, y) for x, y in data["level"]["entities"]["coin"]]
            [self.addCoinBrick(x, y) for x, y in data["level"]["entities"]["coinBrick"]]
            [self.addRandomBox(x, y, item) for x, y, item in data["level"]["entities"]["RandomBox"]]
        except KeyError:
            pass

    def loadLayers(self, data):
        layers = []
        for x in range(*data["level"]["layers"]["sky"]["x"]):
            layers.append(
                (
                    [Tile(self.sprites.spriteCollection.get("sky"), None)
                     for y in range(*data["level"]["layers"]["sky"]["y"])]
                    + [Tile(
                        self.sprites.spriteCollection.get("ground"),
                        pygame.Rect(x * 32, (y - 1) * 32, 32, 32),
                    )
                        for y in range(*data["level"]["layers"]["ground"]["y"])]
                )
            )
        self.level = list(map(list, zip(*layers)))

    def loadObjects(self, data):
        for x, y in data["level"]["objects"]["bush"]:
            self.addBushSprite(x, y)
        for x, y in data["level"]["objects"]["cloud"]:
            self.addCloudSprite(x, y)
        for x, y, z in data["level"]["objects"]["pipe"]:
            self.addPipeSprite(x, y, z)
        for x, y in data["level"]["objects"]["sky"]:
            self.level[y][x] = Tile(self.sprites.spriteCollection.get("sky"), None)
        for x, y in data["level"]["objects"]["ground"]:
            self.level[y][x] = Tile(
                self.sprites.spriteCollection.get("ground"),
                pygame.Rect(x * 32, y * 32, 32, 32),
            )

    def addCloudSprite(self, x, y):
        try:
            for yOff in range(0, 2):
                for xOff in range(0, 3):
                    self.level[y + yOff][x + xOff] = Tile(
                        self.sprites.spriteCollection.get(f"cloud{yOff + 1}_{xOff + 1}"), None)
        except IndexError:
            return

    def addPipeSprite(self, x, y, length=2):
        try:
            self.level[y][x] = Tile(
                self.sprites.spriteCollection.get("pipeL"),
                pygame.Rect(x * 32, y * 32, 32, 32),
            )
            self.level[y][x + 1] = Tile(
                self.sprites.spriteCollection.get("pipeR"),
                pygame.Rect((x + 1) * 32, y * 32, 32, 32),
            )
            for i in range(1, length + 20):
                self.level[y + i][x] = Tile(
                    self.sprites.spriteCollection.get("pipe2L"),
                    pygame.Rect(x * 32, (y + i) * 32, 32, 32),
                )
                self.level[y + i][x + 1] = Tile(
                    self.sprites.spriteCollection.get("pipe2R"),
                    pygame.Rect((x + 1) * 32, (y + i) * 32, 32, 32),
                )
        except IndexError:
            return

    def addBushSprite(self, x, y):
        try:
            self.level[y][x] = Tile(self.sprites.spriteCollection.get("bush_1"), None)
            self.level[y][x + 1] = Tile(self.sprites.spriteCollection.get("bush_2"), None)
            self.level[y][x + 2] = Tile(self.sprites.spriteCollection.get("bush_3"), None)
        except IndexError:
            return

    def addCoinBox(self, x, y):
        self.level[y][x] = Tile(None, pygame.Rect(x * 32, y * 32 - 1, 32, 32))
        self.entityList.append(
            CoinBox(
                self.screen,
                self.sprites.spriteCollection,
                x,
                y,
                self.sound,
                self.dashboard,
            )
        )

    def addRandomBox(self, x, y, item):
        self.level[y][x] = Tile(None, pygame.Rect(x * 32, y * 32 - 1, 32, 32))
        self.entityList.append(
            RandomBox(
                self.screen,
                self.sprites.spriteCollection,
                x,
                y,
                item,
                self.sound,
                self.dashboard,
                self
            )
        )

    def addCoin(self, x, y):
        self.entityList.append(Coin(self.screen, self.sprites.spriteCollection, x, y))

    def addCoinBrick(self, x, y):
        self.level[y][x] = Tile(None, pygame.Rect(x * 32, y * 32 - 1, 32, 32))
        self.entityList.append(
            CoinBrick(
                self.screen,
                self.sprites.spriteCollection,
                x,
                y,
                self.sound,
                self.dashboard
            )
        )
    def addGoomba(self, x, y):
        if y < len(self.level) and x < len(self.level[0]):
            self.entityList.append(
                Goomba(self.screen, self.sprites.spriteCollection, x, y, self, self.sound)
            )
            # self.logger.debug(f"Added Goomba at ({x}, {y}) - Entity list: {self.entityList}")

    def addKoopa(self, x, y):
        if y < len(self.level) and x < len(self.level[0]):
            self.entityList.append(
                Koopa(self.screen, self.sprites.spriteCollection, x, y, self, self.sound)
            )
            # self.logger.debug(f"Added Koopa at ({x}, {y}) - Entity list: {self.entityList}")

    def addRedMushroom(self, x, y):
        self.entityList.append(
            RedMushroom(self.screen, self.sprites.spriteCollection, x, y, self, self.sound)
        )

    def add_items(self):
        if self.n_extention % 5 == 0:
            dominant_emotion = self.sentiment_analyzer.get_dominant_emotion()
            self.add_item_based_on_emotion(dominant_emotion)

    def add_item_based_on_emotion(self, emotion):
        x = self.levelLength - 1
        y = 8
        if emotion == 'neutral':
            self.addKoopa(x, y)
        elif emotion == 'happy':
            self.addGoomba(x, y)
        elif emotion == 'sad':
            self.addRandomBox(x, 10, 'Coin')
        elif emotion == 'angry':
            self.addRandomBox(x, 10, 'RedMushroom')

    def drawLevel(self, camera):
        for y in range(0, 15):
            for x in range(0 - int(camera.pos.x + 1), 20 - int(camera.pos.x - 1)):
                if self.level[y][x].sprite is not None:
                    if self.level[y][x].sprite.redrawBackground:
                        self.screen.blit(
                            self.sprites.spriteCollection.get("sky").image,
                            ((x + camera.pos.x) * 32, y * 32),
                        )
                    self.level[y][x].sprite.drawSprite(
                        x + camera.pos.x, y, self.screen
                    )
        self.updateEntities(camera)
        self.extend_level()

    def extend_level(self):
        current_length = len(self.level[0])
        extension = self.levelLength - current_length
        if extension > 0:
            for i in range(extension):
                new_sky_layer = [
                    Tile(self.sprites.spriteCollection.get("sky"), None)
                    for y in range(0, 13)
                ]
                new_ground_layer = [
                    Tile(
                        self.sprites.spriteCollection.get("ground"),
                        pygame.Rect((current_length + i) * 32, (y-1) * 32, 32, 32),
                    )
                    for y in range(14, 16)
                ]
                new_column = new_sky_layer + new_ground_layer
                for row, tile in enumerate(new_column):
                    self.level[row].append(tile)
            self.n_extention += extension
            self.add_items()

    def updateEntities(self, cam):
        for entity in self.entityList:
            # self.logger.debug(f"Updating entity {entity} at position {entity.rect.topleft}")
            # self.logger.debug(f"Before update - alive: {entity.alive}, active: {entity.active}, bouncing: {entity.bouncing}")
            entity.update(cam)
            # self.logger.debug(f"After update - alive: {entity.alive}, active: {entity.active}, bouncing: {entity.bouncing}")
            
            # Ne pas supprimer les entités dont l'état vivant est None directement
            if entity.alive is None:
                # self.logger.debug(f"Entity {entity} is marked for removal from entityList")
                entity.to_remove = True  # Marquer l'entité pour suppression
                    
        # Supprimer les entités marquées pour suppression après l'itération
        self.entityList = [entity for entity in self.entityList if not getattr(entity, 'to_remove', False)]
        # self.logger.debug(f"Entity list after update: {self.entityList}")

    def update(self, camera):
        self.frame += 1
        self.sentiment_analyzer.capture_emotions_continuously()
        now_time = dt.now() - self.start_time
        now_time = str(now_time).split(".")[0]
        self.max_pos_cam = max(self.max_pos_cam, abs(round(camera.pos.x)))
        new_level_length = self.initial_length + self.max_pos_cam
        self.levelLength = max(self.levelLength, new_level_length)
        self.drawLevel(camera)