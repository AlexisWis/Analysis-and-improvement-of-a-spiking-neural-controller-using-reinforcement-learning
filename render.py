import pygame as pg
from typing import Tuple
import numpy as np
from action_enum import AgentAction

#Renders the scene. IMPORTANT: Because ipycanvas uses the html canvas coordinates, the y-axis is inverted.
class Renderer():
    def __init__(self, width: int, height: int, origin_x: int = 0, origin_y: int = 0, SCALE: int = 1) -> None:
        self.width = width
        self.height = height
        self.origin = (origin_x, origin_y)
        self.SCALE = SCALE #1m = SCALE pixels

        pg.display.init()
        pg.display.set_caption("Pole Balancing Simulator")
        pg.font.init()
        self.screen = pg.display.set_mode((width, height))
    
    #Translates global coordinates into screen coordinates
    def translate(self, x: int, y: int) -> Tuple[int, int]:
        return (x+self.origin[0], -y+self.origin[1])
    
    #Draws ground. offset is there to shift the ground below the car
    def draw_ground(self, offset: int, color) -> None:
        ground = pg.Rect(self.translate(-self.width//2, -offset * self.SCALE), (self.width, self.height-self.origin[1]-offset * self.SCALE))
        pg.draw.rect(self.screen, color, ground)

    #Draws car. pos_y is omitted because the car's center should be at y = 0
    def draw_car(self, pos_x: float, car_color = "blue", wheel_color = "black") -> None:
        pos_x *= self.SCALE
        #values, hard-coded for now, in meters
        width = 0.5 * self.SCALE
        height = 0.25 * self.SCALE
        wheel_radius = 0.1 * self.SCALE

        car_body = pg.Rect(self.translate(pos_x - width/2, height/2), (width, height))
        pg.draw.rect(self.screen, car_color, car_body)
        pg.draw.circle(self.screen, wheel_color, 
                           self.translate(pos_x - width/2 + wheel_radius, -height/2), wheel_radius)
        pg.draw.circle(self.screen, wheel_color, 
                           self.translate(pos_x + width/2 - wheel_radius, -height/2), wheel_radius)

    #Draws the pole
    def draw_pole(self, pos_x: float, theta: float, length: float, width: float = 0.1, color = "red") -> None:
        pos_x *= self.SCALE
        width = int(width * self.SCALE)
        pole_end_x = length * np.sin(theta) * self.SCALE + pos_x
        pole_end_y = length * np.cos(theta) * self.SCALE
        pg.draw.line(self.screen, color, self.translate(pos_x, 0), self.translate(pole_end_x, pole_end_y), width)

    #Clears the entire canvas
    def draw_clear(self) -> None:
        self.screen.fill("white")

    #Draws physical values
    def draw_stats(self, theta: float, w: float, v: float, x: float, 
                    episode: int, 
                    spikes_left : int, spikes_right : int, 
                    action: int) -> None:
        font = pg.font.Font(None, 24)
        #Physics stats, drawn left
        text = "angle: " + str(theta)[:4] + \
            "\nangular velocity: " + str(w)[:4] + \
            "\nposition: " + str(x)[:4] + \
            "\nvelocity" + str(v)[:4] + \
            " \nepisode: " + str(episode)
        lines = text.split('\n')
        y_pos = 10
        for line in lines:
            text_surface = font.render(line, True, (0,0,0))
            self.screen.blit(text_surface, (10, y_pos))
            y_pos += 30

        #Network stats, drawn right
        text = "Spikes left: " + str(spikes_left) + \
            "\nSpikes right: " + str(spikes_right)[:4] + \
            "\nTaken action: " + ("Left" if action==AgentAction.LEFT else "Right" if action==AgentAction.RIGHT else "Failure")
        lines = text.split('\n')
        y_pos = 10
        for line in lines:
            text_surface = font.render(line, True, (0,0,0))
            self.screen.blit(text_surface, (self.width - 200, y_pos))
            y_pos += 30
    
    def display(self) -> None:
        pg.display.flip()