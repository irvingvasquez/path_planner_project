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
import numpy.linalg as LA

#%matplotlib inline

def extract_polygons_and_centers(data):

    polygons = []
    centers = []
 
    safe_distance = 5

    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        
        # Extract the 4 corners of the obstacle
        # 
        obstacle = [north - d_north - safe_distance, north + d_north + safe_distance, east - d_east - safe_distance, east + d_east + safe_distance]
        # NOTE: The order of the points matters since
        # `shapely` draws the sequentially from point to point.
        #
        # If the area of the polygon is 0 you've likely got a weird
        # order.
        corners = [(obstacle[0], obstacle[2]), (obstacle[0], obstacle[3]), (obstacle[1], obstacle[3]), (obstacle[1], obstacle[2])]
        
        # Compute the height of the polygon
        height = alt + d_alt + safe_distance

        poly = Polygon(corners)
        polygons.append((poly, height))

        c = [north, east]
        centers.append(c)
    return polygons, np.asarray(centers)


def collides(polygons, tree, point):   
    # Determine whether the point collides with any obstacles.
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
    # sample points randomly
    xmin = np.min(data[:, 0] - data[:, 3])
    xmax = np.max(data[:, 0] + data[:, 3])

    ymin = np.min(data[:, 1] - data[:, 4])
    ymax = np.max(data[:, 1] + data[:, 4])

    zmin = 5
    zmax = 5

    print("X")
    print("min = {0}, max = {1}\n".format(xmin, xmax))

    print("Y")
    print("min = {0}, max = {1}\n".format(ymin, ymax))

    print("Z")
    print("min = {0}, max = {1}".format(zmin, zmax))

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
    print('Connecting milestones...')
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

def closest_point(graph, current_point):
    closest_point = None
    dist = 100000
    for p in graph.nodes:
        d = LA.norm(np.array(p) - np.array(current_point))
        if(d < dist):
            closest_point = p
            dist = d
    return closest_point

def construct_prm(data):
    print('Constructing the PRM...')
    num_samples = 3000
    k = 7 #NN

    # Create the centers of the polygons
    obstacles, centers = extract_polygons_and_centers(data)
    #print(centers[:10])
    
    # then use KDTree to find nearest neighbor polygon
    # and test for collision
    obst_tree = KDTree(centers)

    milestones = get_milestones(data, obstacles, obst_tree, num_samples)

    # create the graph
    g = create_graph(milestones, obstacles, k)
    
    return g, milestones

def display_grid(data, g, milestones):
    plt.rcParams['figure.figsize'] = 15, 15
    # Create a grid map of the world
    # This will create a grid map at 1 m above ground level
    grid = create_grid(data, 5, 5)

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
        plt.scatter(n1[1] - emin, n1[0] - nmin, c='red')
    
    # Draw connected nodes in green
    for n1 in g.nodes:
        plt.scatter(n1[1] - emin, n1[0] - nmin, c='green')
    
    plt.xlabel('NORTH')
    plt.ylabel('EAST')

    plt.show()

def display_path(data, g, path):
    fig = plt.figure()
    # Create a grid map of the world
    # This will create a grid map at 1 m above ground level
    grid = create_grid(data, 5, 5)


    plt.imshow(grid, cmap='Greys', origin='lower')

    nmin = np.min(data[:, 0])
    emin = np.min(data[:, 1])
   
    # draw nodes
    #for n1 in g.nodes:
    #    plt.scatter(n1[1] - emin, n1[0] - nmin, c='red')
    
    # draw edges
    #for (n1, n2) in g.edges:
    #    plt.plot([n1[1] - emin, n2[1] - emin], [n1[0] - nmin, n2[0] - nmin], 'grey')
    
    # add code to visualize the path
    path_pairs = zip(path[:-1], path[1:])
    for (n1, n2) in path_pairs:
        plt.plot([n1[1] - emin, n2[1] - emin], [n1[0] - nmin, n2[0] - nmin], 'red')
    
    plt.xlabel('NORTH')
    plt.ylabel('EAST')

    plt.show()


def heuristic(n1, n2):
    return LA.norm(np.array(n2) - np.array(n1))



def a_star(graph, heuristic, start, goal):
    """Modified A* to work with NetworkX graphs."""

    path = []
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False
    
    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]

        if current_node == goal:        
            print('PRM Found a path.')
            found = True
            break
        else:
            for next_node in graph[current_node]:
                cost = graph.edges[current_node, next_node]['weight']
                new_cost = current_cost + cost + heuristic(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    queue.put((new_cost, next_node))
                    
                    branch[next_node] = (new_cost, current_node)
             
    path = []
    path_cost = 0
    if found:
        
        # retrace steps
        path = []
        n = goal
        path_cost = branch[n][0]
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        path_cost = -1
    return path[::-1], path_cost
