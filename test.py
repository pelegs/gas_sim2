#!/usr/bin/env python3
# -*- coding: iso-8859-15 -*-


Nx, Ny = 4, 4
x, y = 2, 2
neighbors = [(i, j) for i in range(x-1, x+2) if 0 <= i < Nx
                    for j in range(y-1, y+2) if 0 <= j < Ny]
print(neighbors)
