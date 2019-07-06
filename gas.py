#!/usr/bin/env python3
# -*- coding: iso-8859-15 -*-

import sys
import numpy as np
from numpy import pi, sin, cos, sqrt
import pygame


def normalize(vec):
    L = np.linalg.norm(vec)
    if L != 0:
        return vec/L
    else:
        return vec*0.0

def scale_vec(vec, size):
    new_vec = normalize(vec)
    return new_vec * size

def rotate(vec, angle):
    c = cos(angle)
    s = sin(angle)
    mat = np.array([[c, -s],
                    [s,  c]])
    return np.dot(mat, vec)

def intersection(x1, x2, x3, x4,
                 y1, y2, y3, y4):
    a = ((y3-y4)*(x1-x3) + (x4-x3)*(y1-y3))
    b = ((y1-y2)*(x1-x3) + (x2-x1)*(y1-y3))
    c = ((x4-x3)*(y1-y2) - (x1-x2)*(y4-y3))
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    if c != 0.0:
        return a/c, b/c, p1 + (p2-p1)*(a/c)
    else:
        return 0, 0, np.zeros(2)

def dist(v1, v2):
    return np.linalg.norm(v2-v1)


class wall:
    def __init__(self, start, end, width=1, color=[255]*3):
        self.start = start
        self.end = end
        self.width = width
        self.color = color

        # Additional wall properties
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        self.dir = normalize(np.array([dx, dy]))
        self.angle = np.arctan2(dy, dx)
        self.normal = rotate(self.dir, pi/2)
        self.length = np.linalg.norm(np.array([dx, dy]))
        self.vec = self.dir * self.length
        self.center = (self.start + self.end) / 2

    def set_pos(self, pos):
        self.start = pos - self.vec/2
        self.end = pos + self.vec/2

    def draw(self, surface):
        pygame.draw.line(surface, self.color,
                         self.start.astype(int),
                         self.end.astype(int),
                         self.width)

    def wall_intersection(self, w2):
        return intersection(self.start[0], self.end[0],
                            w2.start[0], w2.end[0],
                            self.start[1], self.end[1],
                            w2.start[1], w2.end[1])



class particle:
    def __init__(self, pos, vel, mass, radius, color):
        self.pos = pos
        self.vel = vel
        self.mass = mass
        self.radius = radius
        self.color = color

    def add_acceleration(self, a, dt):
        self.vel += a*dt

    def move(self, dt):
        self.pos += self.vel * dt

    def draw(self, surface):
        pygame.draw.circle(surface, self.color,
                           self.pos.astype(int),
                           self.radius)

    def wall_collision(self, w, dt):
        next_pos = self.pos + self.vel*dt + scale_vec(self.vel, self.radius)
        ta, tb, intersection_point = intersection(self.pos[0], next_pos[0],
                                                  w.start[0], w.end[0],
                                                  self.pos[1], next_pos[1],
                                                  w.start[1], w.end[1])
        if 0 <= ta <= 1 and 0 <= tb <= 1:
            self.vel = self.vel - 2 * (np.dot(self.vel, w.normal)) * w.normal


def particle_collision(p1, p2):
    distance = dist(p1.pos, p2.pos)
    if distance <= p1.radius + p2.radius:
        dr = p2.pos - p1.pos
        dv = p2.vel - p1.vel
        dR = np.dot(dv, dr) / np.dot(dr, dr) * dr
        M = p1.mass + p2.mass
        overlap = p1.radius + p2.radius - distance
        if overlap > 0.0:
            p2.pos += normalize(dr) * overlap
        p1.vel += 2*p2.mass/M * dR
        p2.vel -= 2*p1.mass/M * dR


s = 800
center = np.array([s/2, s/2])
pygame.display.init()
screen = pygame.display.set_mode((s, s))
pygame.display.flip()

dt = 0.1
gravity = np.array([0, 2])

w1 = wall(start=np.array([s/2-200, s/2-200]),
          end=np.array([s/2+200, s/2-200]),
          width=4,
          color=[255,0,255])
w2 = wall(start=np.array([s/2-200, s/2-200]),
          end=np.array([s/2-200, s/2+200]),
          width=4,
          color=[255,0,255])
w3 = wall(start=np.array([s/2+200, s/2+200]),
          end=np.array([s/2-200, s/2+200]),
          width=4,
          color=[255,0,255])
w4 = wall(start=np.array([s/2+200, s/2+200]),
          end=np.array([s/2+200, s/2-200]),
          width=4,
          color=[255,0,255])

balls1 = [particle(pos=np.random.uniform(s/2-100, s/2+100, 2),
                   vel=np.random.uniform(-10, 10, size=2),
                   mass=1,
                   radius=5,
                   color=[0,255,0])
         for _ in range(10)]
balls2 = [particle(pos=np.random.uniform(s/2-100, s/2+100, 2),
                   vel=np.random.uniform(-10, 10, size=2),
                   mass=10,
                   radius=10,
                   color=[255,0,100])
         for _ in range(20)]
balls = balls1 + balls2

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                pygame.quit()
                sys.exit()

    # Physics
    for b1 in balls:
        b1.add_acceleration(gravity, dt)
        b1.move(dt)
        b1.wall_collision(w1, dt)
        b1.wall_collision(w2, dt)
        b1.wall_collision(w3, dt)
        b1.wall_collision(w4, dt)
        for b2 in balls:
            if b1 is not b2:
                particle_collision(b1, b2)

    # Drawing
    screen.fill(3*[0])
    w1.draw(screen)
    w2.draw(screen)
    w3.draw(screen)
    w4.draw(screen)
    for b in balls:
        b.draw(screen)
    pygame.display.update()

print('')
