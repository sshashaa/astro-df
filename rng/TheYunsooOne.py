# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 16:51:30 2022

@author: wesle
"""
import numpy as np
#from mrg32k3a import MRG32k3a
import random as myrng
import math

s = 1
ss = 10
sss = 15
print('Seed', s, ss, sss)
#myrng = MRG32k3a(s_ss_sss_index=[s, ss, sss])
myrng.seed(2)
T = 100
numintersections = 4
interval = 10
lam = 1
distance = 5
speed = 2.5
carlength = 1
reaction = 0.1
offset = [0, 10, 5, 7]

outbounds = T + 1
t = 0
nextcargen = 0
carSimIndex = 0
nextStart = outbounds
nextSecArrival = outbounds
minPrimArrival = 0

from TrafficLightClasses import Road
from TrafficLightClasses import Intersection
from TrafficLightClasses import Car

# Draw out map of all locations in system
roadmap = np.array([
    ['', 'N1', 'N2', ''],
    ['W1', 'A', 'B', 'E1'],
    ['W2', 'C', 'D', 'E2'],
    ['', 'S1', 'S2', '']])
# List each location and the locations that are next accessible
graph = {'A': ['N1', 'B', 'C'],
         'B': ['N2', 'E1', 'D'],
         'C': ['A', 'S1', 'W2'],
         'D': ['C', 'B', 'S2'],
         'N1': ['A'],
         'N2': ['B'],
         'E1': [],
         'E2': ['D'],
         'S1': ['C'],
         'S2': ['D'],
         'W1': ['A'],
         'W2': []
         }
# Lists each location in the system
points = list(graph.keys())


# Find the shortest path between two points
def find_shortest_path(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path

    shortest = None
    for node in graph[start]:
        if node not in path:
            newpath = find_shortest_path(graph, node, end, path)
            if newpath:
                if not shortest or len(newpath) < len(shortest):
                    shortest = newpath
    return shortest


# Generates shortest path through two random start and end locations
def generate_path():
    start = myrng.randint(numintersections, 11)
    end = myrng.randint(numintersections, 11)
    path = find_shortest_path(graph, points[start], points[end])
    return path


'''
ROADS
'''


# Takes in road and finds its direction based on the map
def find_direction(start, end, roadmap):
    yloc1, xloc1 = np.where(roadmap == start)
    yloc2, xloc2 = np.where(roadmap == end)
    if xloc1 > xloc2:
        direction = 'West'
    elif xloc1 < xloc2:
        direction = 'East'
    elif yloc1 > yloc2:
        direction = 'North'
    else:
        direction = 'South'
    return direction


# Assigns the direction of a turn when given two roads
def find_turn(roadcombo):
    turnkey = {'Straight': ['WestWest', 'EastEast', 'SouthSouth', 'NorthNorth'],
               'Left': ['NorthWest', 'EastNorth', 'SouthEast', 'WestSouth'],
               'Right': ['NorthEast', 'WestNorth', 'SouthWest', 'EastSouth']
               }
    turn = ''
    for key, values in turnkey.items():
        for value in values:
            if roadcombo == value:
                turn = key
    return turn


# Generates list of all road objects in the system
roads = list()
roadid = 0
for key, value in graph.items():
    for value in graph[key]:
        direction = find_direction(key, value, roadmap)
        roads.append(Road(roadid, key, value, direction))
        roads[roadid].queue.append(0)
        print('Road', roadid, ':', key, value, direction)
        roadid += 1


# Finds the roads that a car will take on its path
def find_roads(visits):
    path = list()
    for i in range(len(visits) - 1):
        for road in roads:
            if road.startpoint == visits[i] and road.endpoint == visits[i + 1]:
                path.append(road)
    return path


'''
INTERSECTIONS
'''


# Generates light schedule based on a given interval
def gen_lightschedule(interval, intersection, index):
    schedule = list()
    '''
    startrow = np.array(np.where(roadmap == 'A'))[0]
    startcol = np.array(np.where(roadmap == 'A'))[1]
    endrow = np.array(np.where(roadmap == location))[0]
    endcol = np.array(np.where(roadmap == location))[1]
    distance = int((endrow - startrow) + (endcol - startcol))
    '''
    intersection.offset = offset[index]
    offsetcalc = (intersection.offset % interval)
    if offsetcalc == 0:
        offsetcalc = interval
    for i in range(math.ceil(T / interval) + 1):
        if i == 0:
            schedule.append(0)
        else:
            schedule.append(offsetcalc + (interval * (i - 1)))
    return schedule


# Generates list of all intersection objects
intersections = list()
for i in range(numintersections):
    location = points[i]
    intersections.append(Intersection(location, roads))
    schedule = gen_lightschedule(interval, intersections[i], i)
    intersections[i].schedule = schedule
    print('Intersection', intersections[i].name, ':', schedule)
    intersections[i].connect_roads(roads)


# Finds the next time any intersection light will change signal
def find_nextlightchange(intersections, t):
    mintimechange = T
    location = []
    for intersection in intersections:
        index = 0
        while index < len(intersection.schedule) and t >= intersection.schedule[index]:
            index += 1
        if index != len(intersection.schedule) and intersection.schedule[index] <= mintimechange:
            mintimechange = intersection.schedule[index]
            location.append(intersection)
    for intersection in location:
        for road in intersection.horizontalroads:
            road.nextstart = mintimechange
        for road in intersection.verticalroads:
            road.nextstart = mintimechange

    return mintimechange, location


# Updates the intersections with their new light status
def update_intersections(t, intersections):
    print("I AM CHANGING A LIGHT AT TIME:", t)
    if t == 0:
        nextlightlocation = intersections
    else:
        nextlightlocation = []
        for intersection in intersections:
            if t in intersection.schedule:
                nextlightlocation.append(intersection)
    for intersection in nextlightlocation:
        for road in intersection.horizontalroads:
            road.update_light(intersection.schedule, t)
            print('Road', road.roadid, 'is now:', road.status)
            road.nextchange = t + interval
        for road in intersection.verticalroads:
            road.update_light(intersection.schedule, t)
            print('Road', road.roadid, 'is now:', road.status)
            road.nextchange = t + interval


'''
CAR Functions
'''
# Generates list of all car objects as they are created
cars = list()


def gen_car(initialarrival):
    visits = generate_path()
    while visits == None or len(visits) == 1:
        visits = generate_path()
    identify = len(cars)
    path = find_roads(visits)
    cars.append(Car(identify, initialarrival, path))
    cars[identify].nextstart = outbounds
    cars[identify].nextSecArrival = outbounds
    print('Car', identify, ':', visits)


# Finds a car's place in queue and assigns it a new start time
def find_place_in_queue(car, road, t):
    queueindex = len(road.queue) - 1
    while road.queue[queueindex] == 0 and queueindex > 0:
        queueindex -= 1
    # Car is not the first in its queue
    if queueindex != 0 or road.queue[0] != 0:
        # Car is second in queue
        if len(road.queue) == queueindex + 1:
            road.queue.append(car)
        # Car is third or later in queue
        else:
            road.queue[queueindex + 1] = car
        car.placeInQueue = queueindex + 1
        car.nextstart = road.queue[queueindex].nextstart + reaction
    # Car is the first in its queue
    else:
        road.queue[queueindex] = car
        car.placeInQueue = queueindex
        # Car is at the end of its path
        if car.locationindex == len(car.path) - 1:
            car.nextstart = outbounds
            car.nextSecArrival = outbounds
        # Car still has a road to travel to
        else:
            # Light is green on the road that the car is on
            if road.status == True:
                car.nextstart = t
            # Light is red on the road that the car is on
            else:
                car.nextstart = road.nextchange


# Lights are turned on and the first car is created
update_intersections(0, intersections)
gen_car(nextcargen)
currentcar = cars[carSimIndex]
movingcar = cars[0]
arrivingcar = movingcar

# Loops through time until runtime is reached
while t < T:
    # Assigns the next time a light changes
    nextLightTime = find_nextlightchange(intersections, t)[0]
    # The next event is a car being introduced to the system
    if min(minPrimArrival, nextLightTime, nextStart, nextSecArrival, nextcargen) == minPrimArrival:
        t = minPrimArrival
        cars[carSimIndex].prevstop = t
        print('Car', cars[carSimIndex].identify, 'is arriving first at time', t)
        # A new car is generated
        nextcargen = t + myrng.expovariate(lam)
        minPrimArrival = nextcargen
        carSimIndex += 1
        gen_car(nextcargen)

        # The arriving car arrives into the system based on its path
        currentcar = cars[carSimIndex - 1]
        initroad = currentcar.path[currentcar.locationindex]
        find_place_in_queue(currentcar, initroad, t)

    # The next event is a light changing
    elif min(minPrimArrival, nextLightTime, nextStart, nextSecArrival, nextcargen) == nextLightTime:
        t = nextLightTime
        # Intersections that change lights at this time are updated
        update_intersections(t, intersections)

    # The next event is a car starting to move
    elif min(minPrimArrival, nextLightTime, nextStart, nextSecArrival, nextcargen) == nextStart:
        t = nextStart
        print('Time:', t, 'Car', movingcar.identify, 'is starting from road',
              movingcar.path[movingcar.locationindex].roadid, 'at spot', movingcar.placeInQueue)
        # Car is the first in its queue
        if movingcar.placeInQueue == 0:
            # Car's next arrival is set
            movingcar.nextSecArrival = t + (distance / speed)
        # Car is not the first in its queue
        else:
            # Car's next arrival time is set
            movingcar.nextSecArrival = t + (carlength / speed)

        # Car leaves its current queue and is 'moving'
        movingcar.path[movingcar.locationindex].queue[movingcar.placeInQueue] = 0
        movingcar.moving = True
        movingcar.timewaiting = movingcar.timewaiting + (t - movingcar.prevstop)
        movingcar.nextstart = outbounds
        nextStart = outbounds

    # The next event is a car arriving within the system
    elif min(minPrimArrival, nextLightTime, nextStart, nextSecArrival, nextcargen) == nextSecArrival:
        t = nextSecArrival

        # Car is first in its queue
        if arrivingcar.placeInQueue == 0:
            # Car changes the road it is traveling on
            arrivingcar.update_location()
            currentroad = arrivingcar.path[arrivingcar.locationindex]
            # Car is assigned its location and given a new start time
            find_place_in_queue(arrivingcar, currentroad, t)
        # Car is not the first in its queue
        else:
            # Car moves up in its queue
            currentroad = arrivingcar.path[arrivingcar.locationindex]
            currentroad.queue[arrivingcar.placeInQueue] = 0
            currentroad.queue[arrivingcar.placeInQueue - 1] = arrivingcar
            arrivingcar.placeInQueue -= 1
            # Current road has a green light
            if currentroad.status == True:
                arrivingcar.nextstart = t
            # Current road has a red light
            else:
                arrivingcar.nextstart = currentroad.nextchange
        print('Time:', t, 'Car', arrivingcar.identify, 'is arriving at road',
              arrivingcar.path[arrivingcar.locationindex].roadid, 'at spot', arrivingcar.placeInQueue)
        # Car is no longer 'moving'
        movingcar.moving = False
        arrivingcar.nextSecArrival = outbounds
        nextSecArrival = outbounds
        arrivingcar.prevstop = t

    carindex = 0
    minSecArrival = outbounds
    minStart = outbounds
    # Finds the next car to start moving and the next car to arrive
    while carindex < len(cars) - 1:
        testcar = cars[carindex]
        # Car is elligible to be the next starting car
        if min(nextStart, testcar.nextstart) == testcar.nextstart and testcar.nextstart != outbounds:
            minStart = testcar.nextstart
            movingcar = testcar
        # Car is elligible to be the next arriving car
        if min(minSecArrival, testcar.nextSecArrival) == testcar.nextSecArrival and testcar.nextSecArrival != outbounds:
            minSecArrival = testcar.nextSecArrival
            arrivingcar = testcar
        # Next car is tested and the next events are set
        carindex += 1
        nextSecArrival = minSecArrival
        nextStart = minStart

for car in cars:
    if car.locationindex == len(car.path) - 1:
        print('Car', car.identify, 'waiting time:', car.timewaiting)

