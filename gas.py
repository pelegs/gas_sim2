#!/usr/bin/env python3
# -*- coding: iso-8859-15 -*-

import sys
import numpy as np
from numpy import pi, sin, cos, sqrt
import pygame


RED, GREEN, BLUE = 0, 1, 2

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


class Wall:
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



class Particle:
    def __init__(self, pos, vel, mass, radius, color):
        self.pos = pos
        self.vel = vel
        self.mass = mass
        self.radius = radius
        self.color = color

        self.cell = (-1, -1)
        self.neighbors = []

    def set_cell(self, x, y):
        self.cell = (x, y)

    def set_neighors(self, grid):
        # Reset current neighbors list
        self.neighbors = []

        # Create new list
        x, y = self.cell
        Nx, Ny = grid.Nx, grid.Ny
        neighbors = [grid.objects[i][j] for i in range(x-1, x+2) if 0 <= i < Nx
                                        for j in range(y-1, y+2) if 0 <= j < Ny]
        self.neighbors = [object for sublist in neighbors
                                 for object in sublist
                                 if object is not self]


    def add_acceleration(self, a, dt):
        self.vel += a*dt

    def drag(self, k):
        self.vel -= self.vel*k

    def move(self, dt):
        self.pos += self.vel * dt

    def get_kinetic_energy(self):
        return 0.5*self.mass*self.vel**2

    def set_kinetic_energy(self, energy):
        self.vel = scale_vec(self.vel, np.sqrt(2*energy/self.mass))

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
            return True
        else:
            return False

    def in_bounds(self, xmin, ymin, xmax, ymax):
        if xmin < self.pos[0] < xmax and ymin < self.pos[1] < ymax:
            return True
        else:
            return False

    def set_color_by_vel(self, max, channel=RED):
        self.color = [0, 0, 0]
        val = int((np.linalg.norm(self.vel)/max) * 255)
        self.color[channel] = val


class Grid:
    def __init__(self,
                 Nx, Ny,
                 Sx, Sy,
                 Ex, Ey):
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = (Ex-Sx)/Nx
        self.Ly = (Ey-Sy)/Ny

        self.density = np.zeros(shape=(Nx, Ny))

        self.reset()

    def reset(self):
        self.objects = [[[] for _ in range(self.Nx)]
                            for _ in range(self.Ny)]

    def add_object(self, p):
        cellx = int(np.floor(p.pos[0]/self.Lx))
        celly = int(np.floor(p.pos[1]/self.Ly))
        self.objects[cellx][celly].append(p)
        p.set_cell(cellx, celly)
        self.density[cellx, celly] += 1


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


def vel_cm(objects):
    velx = np.sum([object.vel[0] for object in objects])
    vely = np.sum([object.vel[1] for object in objects])
    return np.array([velx, vely])



s = 800
center = np.array([s/2, s/2])
pygame.display.init()
screen = pygame.display.set_mode((s, s))
pygame.display.flip()

dt = 1.0
gravity = np.array([0, 0.1])

w1 = Wall(start=np.array([0, 0]),
          end=np.array([0, s]),
          width=4,
          color=[255,255,255])
w2 = Wall(start=np.array([0, 0]),
          end=np.array([s, 0]),
          width=4,
          color=[255,255,255])
w3 = Wall(start=np.array([s, s]),
          end=np.array([0, s]),
          width=4,
          color=[255,0,255])
w4 = Wall(start=np.array([s, s]),
          end=np.array([s, 0]),
          width=4,
          color=[255,255,255])

num_particles = 100
balls = [Particle(pos=np.random.uniform(100, 500, 2),
                  vel=np.zeros(2),
                  mass=1,
                  radius=10,
                  color=[255, 0, 0])
         for _ in range(num_particles)]
balls[-1].pos = np.array([700.0, 700.0])
balls[-1].vel = np.array([-1.0, -1.0])
balls[-1].mass = 1
balls[-1].set_kinetic_energy(50)
balls[-1].color = [255]*3
#Ek = np.sum([b.get_kinetic_energy() for b in balls])
#for b in balls:
#    b.set_kinetic_energy(0/num_particles)
#Ek = []

grid = Grid(50, 50,
            0.0, 0.0,
            800.0, 800.0)

# Data to collect
#bpos = np.empty(2)
#all_vels = np.empty(shape=(2, num_particles))

# Main loop
run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                run = False

    # Place in grid
    grid.reset()
    for b in balls:
        grid.add_object(b)

    # Create neighbor lists
    for b in balls:
        b.set_neighors(grid)

    # Physics
    for b1 in balls:
        bath1 = b1.wall_collision(w1, dt)
        bath2 = b1.wall_collision(w2, dt)
        bath3 = b1.wall_collision(w3, dt)
        bath4 = b1.wall_collision(w4, dt)
        if bath3:
            b1.set_kinetic_energy(10)
        for b2 in b1.neighbors:
            particle_collision(b1, b2)
        #b1.drag(0.01)
        #b1.add_acceleration(gravity, dt)

    # Move
    for b in balls:
        b.move(dt)
        if not b.in_bounds(0, 0, 800, 800):
            #print('deleted object')
            balls.remove(b)

    # Velocities (for coloring)
    #vels = np.array([np.linalg.norm(b.vel) for b in balls])
    #max = np.max(vels)
    #all_vels = np.append(all_vels, vels)

    # Drawing
    screen.fill(3*[0])
    w1.draw(screen)
    w2.draw(screen)
    w3.draw(screen)
    w4.draw(screen)
    for b in balls:
        #b.set_color_by_vel(max, channel=RED)
        b.draw(screen)
    pygame.display.update()

    # Collect data
    #Ek.append(np.sum([b.get_kinetic_energy() for b in balls]))

# Save collected data
#np.save('vels', all_vels)
#np.save('kinteic_energy', np.array(Ek))

# Exit program
pygame.quit()
sys.exit()
