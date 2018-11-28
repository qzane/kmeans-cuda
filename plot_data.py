#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 22:31:17 2018

@author: qzane
"""

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def read_points(fname):
    points = []
    with open(fname) as f:
        while(1):
            tmp = f.readline()
            if tmp == '':
                break
            if ',' in tmp:
                f1,f2 = tmp.split(',')[:2]
                f1,f2 = float(f1), float(f2)
                points.append((f1,f2))
    return np.array(points)


def read_classes(fname):
    classes = []
    with open(fname) as f:
        while(1):
            tmp = f.readline()
            if tmp == '':
                break
            _class = int(tmp)
            classes.append(_class)
            
    return np.array(classes)
    
def plot(points, classes):
    assert(points.shape[0]==classes.shape[0])
    num_classes = classes.max()+1
    
    cmap = plt.get_cmap('jet')
    colors = [cmap(i) for i in np.linspace(0, 1, num_classes)]
    
    for i in range(num_classes):
        plt.plot(points[classes==i,0], points[classes==i,1], 'x', color=colors[i])
    plt.show()
    


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument('-p', '--points', action='store', type=str, required=True,
                        help='points.txt')
    
    parser.add_argument('-c', '--classes', action='store', type=str, required=True,
                        help='classes.txt')
           
    args = parser.parse_args()  
         
    
    points = read_points(args.points)
    classes = read_classes(args.classes)
    plot(points, classes)