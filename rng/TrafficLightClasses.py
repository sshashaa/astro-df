# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 15:13:19 2022

@author: wesle
"""

class Road:
    def __init__(self, roadid, startpoint, endpoint, direction):
        self.roadid = roadid
        self.startpoint = startpoint
        self.endpoint = endpoint
        self.direction = direction
        self.queue = list()
        self.status = False

    def update_light(self, schedule, t):
        for time in schedule:
            if time == t:
                if self.status == True:
                    self.status = False
                else:
                    self.status = True
                    if len(self.queue) > 0 and self.queue[0] != 0:
                        self.queue[0].starttime = t
    
    def get_data(self):
        return(self.roadid, self.startpoint, self.endpoint, self.direction, self.queue)
        
class Intersection:
    def __init__(self, name, schedule, roads):
        self.name = name
        self.schedule = schedule
        self.horizontalroads = []
        self.verticalroads = []
        
    def connect_roads(self, roads):
        for Road in roads:
            if Road.endpoint == self.name:
                direction = Road.direction
                if direction == 'East' or direction == 'West':
                    self.horizontalroads.append(Road)
                    Road.status = True                    
                else:
                    self.verticalroads.append(Road)
                    Road.status = False
    
    def get_data(self):
        return(self.name, self.schedule)
        
class Car:
    def __init__(self, carid, arrival, path):
        self.identify = carid
        self.arrival = arrival
        self.initialarrival = arrival
        self.path = path
        self.locationindex = 0
        self.timewaiting = 0
        self.primarrival = arrival
        self.placeInQueue = None
        self.nextstart = None
        self.moving = False
        self.nextSecArrival = None
        self.prevstop = 0
        
    def update_location(self):
        self.locationindex += 1

        
    def get_data(self):
        return(self.identify, self.arrival, self.path)