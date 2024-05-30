#!/usr/bin/env python
# coding: utf-8

import numpy as np

class car:
    """
    Aleksander Merrill
    Project 5
    Introduction to Machine Learning
    Car class for recurrent learning on three tracks
    Assumes 'Perfect' frictionless world for ease
    """
    
    def __init__(self, x, y):
        """
        initializes Node with int value
        @param: x - starting x location
        @param: y - starting y location
        @Information: direction - starting direction car is facing (0, 1, 2, 3)
           are (up, right, down, left)
        """
        self.x = x
        self.y = y
        self.x_v = 0 #set velocity
        self.y_v = 0
        self.face = 0
    
    def turn(self, way):
        """
        Turns the car, can only turn left or right, not full 180 degrees
        @param way - left or right ('l', or 'r')
        """
        tempx = self.x_v #when turning, conserve x/y momentum with swap
        tempy = self.y_v
        if way == 'l':
            if self.face == 0:
                self.face = 3
                self.x_v = - tempy
                self.y_v = tempx
            elif self.face == 3:
                self.face = 2
                self.x_v = -tempy
                self.y_v = tempx
            elif self.face == 2:
                self.face = 1
                self.x_v = -tempy
                self.y_v = tempx
            else:
                self.face = 0
                self.x_v = -tempy
                self.y_v = tempx
        elif way == 'r':
            if self.face == 0:
                self.face = 1
                self.x_v = tempy
                self.y_v = -tempx
            elif self.face == 3:
                self.face = 0
                self.x_v = tempy
                self.y_v = -tempx
            elif self.face == 2:
                self.face = 3
                self.x_v = tempy
                self.y_v = -tempx
            else:
                self.face = 2
                self.x_v = tempy
                self.y_v = -tempx
    def accel(self, dirs):
        """
        Accelerate the car in a direction on an axis. Positive is up/right
        Can only accelerate in direction by +- 1
        @param dirs - if there is acceleration in x or y direction, + or - or 0 for one or both
        """
        if np.random.rand() <= .8:
            if dirs.__contains__('-x'):
                self.x_v -= 1
            elif dirs.__contains__('x'):
                self.x_v += 1
            if dirs.__contains__('-y'):
                self.y_v -= 1
            elif dirs.__contains__('y'):
                self.y_v += 1
            if self.x_v < -5:
                self.x_v = -5
            elif self.x_v > 5:
                self.x_v = 5
            if self.y_v < -5:
                self.y_v = -5
            elif self.y_v > 5:
                self.y_v = 5
    def go(self):
        """
        Movement at the end of a period
        """
        self.x += self.x_v
        self.y += self.y_v
    
    def getV(self):
        """
        return velocities
        """
        return (self.x_v, self.y_v)
    def getSpot(self):
        """
        Return location
        """
        return (self.x, self.y)
    
    def copy(self):
        truck = car(self.x, self.y)
        truck.x_v = self.x_v
        truck.y_v = self.y_v
        truck.face = self.face
        return truck
