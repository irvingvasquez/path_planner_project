import sys
#!{sys.executable} -m pip install -I networkx==2.1
import pkg_resources
pkg_resources.require("networkx==2.1")
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
from queue import PriorityQueue
from sklearn.neighbors import KDTree
from grid import create_grid

def extract_polygons_and_centers(data):

    polygons = []
    centers = []
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        
        # TODO: Extract the 4 corners of the obstacle
        # 
        obstacle = [north - d_north, north + d_north, east - d_east, east + d_east]
        # NOTE: The order of the points matters since
        # `shapely` draws the sequentially from point to point.
        #
        # If the area of the polygon is 0 you've likely got a weird
        # order.
        corners = [(obstacle[0], obstacle[2]), (obstacle[0], obstacle[3]), (obstacle[1], obstacle[3]), (obstacle[1], obstacle[2])]
        
        # TODO: Compute the height of the polygon
        height = alt + d_alt

        # TODO: Once you've defined corners, define polygons
        #p = Polygon(corners)
        #polygons.append((p, height))
        poly = Polygon(corners)
        polygons.append((poly, height))

        c = [north, east]
        centers.append(c)
    return polygons, np.asarray(centers)


def collides(polygons, tree, point):   
    # TODO: Determine whether the point collides
    # with any obstacles.
    collide = False
    x,y,z = point
    p = Point(x, y)
    
    #find the nearest polygon center
    p_array = np.array([x,y])
    idx = tree.query([p_array], k=1, return_distance = False)[0]
    
    poly = polygons[int(idx)]
    polygon, height = poly
    if polygon.contains(p):
        if z < height:
            collide = True
            
    return collide


def get_milestones(data, polygons, tree, num_samples):
    
    # TODO: sample points randomly

    xmin = np.min(data[:, 0] - data[:, 3])
    xmax = np.max(data[:, 0] + data[:, 3])

    ymin = np.min(data[:, 1] - data[:, 4])
    ymax = np.max(data[:, 1] + data[:, 4])

    zmin = 0
    # Limit the z axis for the visualization
    zmax = 10

    print("X")
    print("min = {0}, max = {1}\n".format(xmin, xmax))

    print("Y")
    print("min = {0}, max = {1}\n".format(ymin, ymax))

    print("Z")
    print("min = {0}, max = {1}".format(zmin, zmax))

    #num_samples = 1000

    xvals = np.random.uniform(xmin, xmax, num_samples)
    yvals = np.random.uniform(ymin, ymax, num_samples)
    zvals = np.random.uniform(zmin, zmax, num_samples)

    samples = list(zip(xvals, yvals, zvals))

    samples[:10]

    to_keep = []
    for point in samples:
        if not collides(polygons, tree, point):
            to_keep.append(point)

    print('feasible samples:', len(to_keep))

    return to_keep

def can_connect(p1, p2, obstacles):
    x1,y1,z1 = p1
    x2,y2,z2 = p2
    
    line = LineString([(x1, y1), (x2, y2)])
    free = True
    
    for obst in obstacles:
        poly, height = obst
        if line.crosses(poly):
            free = False
    
    return free

def create_graph(milestones, polygons, k):
    array_milestones = np.array(milestones)
    #print(array_milestones[:10])
    m_tree = KDTree(array_milestones)
    g = nx.Graph()
    
    for mil in milestones:
        #print(mil)
        
        mil_array = np.array(mil)
        #print(mil_array)
        # find the nearest k milestones
        distances, indexes = m_tree.query([mil_array], k)

        #print(dist)
        #print(idx)
        for dist, idx in zip(distances[0], indexes[0]):
            #print(dist)
            #print(idx)
            if dist > 0:
                if can_connect(mil, milestones[int(idx)], polygons):
                    g.add_edge(mil, milestones[int(idx)], weight=dist)
        
    return g

def construct_prm(data):
    num_samples = 1000
    k = 6 #NN

    # Create the centers of the polygons
    obstacles, centers = extract_polygons_and_centers(data)
    print(centers[:10])
    
    # then use KDTree to find nearest neighbor polygon
    # and test for collision
    obst_tree = KDTree(centers)

    milestones = get_milestones(data, obstacles, obst_tree, num_samples)

    # TODO: create the graph
    k = 6
    g = create_graph(milestones, obstacles, k)
    
    return g, milestones

def display_grid(data, g, milestones):
    # Create a grid map of the world
    # This will create a grid map at 1 m above ground level
    grid = create_grid(data, 1, 1)

    fig = plt.figure()

    plt.imshow(grid, cmap='Greys', origin='lower')

    nmin = np.min(data[:, 0])
    emin = np.min(data[:, 1])

    # If you have a graph called "g" these plots should work
    # Draw edges
    for (n1, n2) in g.edges:
        plt.plot([n1[1] - emin, n2[1] - emin], [n1[0] - nmin, n2[0] - nmin], 'black' , alpha=0.5)

    # Draw all nodes connected or not in blue
    for n1 in milestones:
        plt.scatter(n1[1] - emin, n1[0] - nmin, c='blue')
    
    # Draw connected nodes in red
    for n1 in g.nodes:
        plt.scatter(n1[1] - emin, n1[0] - nmin, c='red')
    
    plt.xlabel('NORTH')
    plt.ylabel('EAST')

    plt.show()
