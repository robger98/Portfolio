#!/usr/bin/env python3

################################################################
##
## GRAPH COLORING PROBLEM GENERATOR
##
## Generates a planar graph.
##
################################################################

import random
import numpy as np
# import pygame


class Point:
    ID_COUNT = 0
    def __init__(self, x, y):
        self.id = Point.ID_COUNT
        Point.ID_COUNT += 1
        self.x = x
        self.y = y

    def transform(self, xt, yt):
        return (xt(self.x), yt(self.y))

    def dist(self, them):
        return abs(self.x - them.x) + abs(self.y - them.y)

    def __repr__(self):
        return "p({:.4f}, {:.4f})".format(self.x, self.y)

## From http://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
def _ccw(A,B,C):
    return (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)

class Line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    @property
    def endpoints(self):
        return [self.p1, self.p2]

    def transform(self, xt = lambda x: x, yt = lambda y: y):
        start = self.p1.transform(xt, yt)
        end = self.p2.transform(xt, yt)
        return (start, end)

    def intersects(self, them):
        A = self.p1
        B = self.p2
        C = them.p1
        D = them.p2
        return _ccw(A,C,D) != _ccw(B,C,D) and _ccw(A,B,C) != _ccw(A,B,D)

    def __repr__(self):
        return "[{} -> {}]".format(self.p1, self.p2)

def _random_point():
    return Point(random.uniform(-10, 10), random.uniform(-10, 10))

def _find_line(x, lines, pairs):
    for i, x1 in enumerate(x[:-1]):
        shortest_items = sorted([(x2, x1.dist(x2)) for x2 in x[i+1:]], key=lambda item: item[1])
        for x2, _ in shortest_items:
            l1 = Line(x1, x2)
            if not (x1, x2) in pairs and not _line_intersects(l1, lines):
                return l1
    return None

def _line_intersects(l1, lines):
    for l2 in lines:
        if l1.p1 in l2.endpoints or l1.p2 in l2.endpoints:
            continue
        if l1.intersects(l2): return True
    return False

def gen(num_points=100):
    x = [_random_point() for _ in range(num_points)]
    lines = set([])
    pairs = set([])
    while True:
        random.shuffle(x)
        line = _find_line(x, lines, pairs)
        if line:
            pairs.add((line.p1, line.p2))
            lines.add(line)
        else:
            break
    return (x, lines)

def draw(x, lines):
    try:
        response = input("Would you like to view the network? [Y/n] ")
        if response.lower() == 'n':
            return
        import pygame
    except ImportError:
        print("Please install pygame ('pip3 install pygame') to view the network.")
        return
    
    pygame.init()
    screen = pygame.display.set_mode((450, 450))
    for line in lines:
        (start, end) = line.transform(xt=lambda x: 25 + int(x * 20 + 200),
                                      yt=lambda y: 25 + int(y * 20 + 200))
        pygame.draw.line(screen, (255, 255, 255), start, end)
    print("Press ESC to exit and save the file.")
    while 1:
        # From https://stackoverflow.com/a/7055453
        for event in pygame.event.get():
           if event.type == pygame.QUIT:
               return
           elif event.type == pygame.KEYDOWN:
               if event.key == pygame.K_ESCAPE:
                   pygame.quit()
                   return
        pygame.display.flip()

if __name__ == '__main__':
    import argparse, json

    parser = argparse.ArgumentParser(description="Generates and displays a randomly generated graph coloring problem.")
    parser.add_argument('num_points', type=int, help="Number of points on the graph to generate.")
    parser.add_argument('--output', type=str, default='gcp.json', help="Where to save the generated network. (default: 'gcp.json')")
    args = parser.parse_args()

    print("Generating a planar graph with {} points...".format(args.num_points))
    (x, lines) = gen(num_points=args.num_points)
    #draw(x, lines)

    print("Writing to '{}'...".format(args.output))
    with open(args.output, 'w') as f:
        f.write(json.dumps({
            'num_points': len(x),
            'points': { p.id: (p.x, p.y) for p in x },
            'edges': [ (line.p1.id, line.p2.id) for line in lines ]
        }, indent=2))
    print("""Done! You can now import the results into a Python script with the code:

import json

with open('{}', r) as f:
    data = json.load(f)
""".format(args.output))
