from collections import defaultdict
from io import DEFAULT_BUFFER_SIZE
import trimesh
import argparse
import numpy as np
import os 
from typing import DefaultDict
from tqdm import tqdm
from trimesh.exchange.load import mesh_formats
from trimesh.graph import neighbors
from itertools import chain

def colfunc(val, minval, maxval, startcolor, stopcolor):
    """ Convert value in the range minval...maxval to a color in the range
        startcolor to stopcolor. The colors passed and the one returned are
        composed of a sequence of N component values (e.g. RGB).
    """
    f = float(val-minval) / (maxval-minval)
    return tuple(f*(b-a)+a for (a, b) in zip(startcolor, stopcolor))

def export_cur2color(file_name, angle_list, points, removed_point):
    RED, YELLOW, GREEN  = (1, 0, 0), (1, 1, 0), (0, 1, 0)
    CYAN, BLUE, MAGENTA = (0, 1, 1), (0, 0, 1), (1, 0, 1)
    minval, maxval = 0.0, 1.0

    f = open(file_name, 'a')
    for point_idx in tqdm(range(len(angle_list))):
        if point_idx in removed_point:
            r,g,b = 0, 0, 0
        else:
            r,g,b = colfunc(angle_list[point_idx]/np.pi,minval, maxval, BLUE, RED)
        for i in range(points.shape[1]):       
            f.write( str(points[point_idx][i]) + ' ')
        f.write(str(r) + ' ' + str(g) + ' ' + str(b) + '\n')
    f.close()


def export_cur(file_name, angle_list, points):
    f = open(file_name, 'a')
    for point_idx in tqdm(range(len(angle_list))):
        for i in range(points.shape[1]):       
            f.write( str(points[point_idx][i]) + ' ')
        f.write(str(angle_list[point_idx]) + '\n')
    f.close()

def get_deleted_points(old_vertices, new_vertices):
    deleted_points = []
    dict = DefaultDict(int)
    for vertice in tqdm(new_vertices):
        dict[str(vertice)] = len(dict)
    for vertice in tqdm(old_vertices):
        if not str(vertice) in dict:
            deleted_points.append(vertice)
    points = trimesh.points.PointCloud(deleted_points)

    return points

# delete_noise:          delete the noise points of the input mesh
# input:
#   @mesh:               a trimesh type mesh
#   @threshold:          if the connected component's point number is less than the thresshold, it will be deleted 
# return:
#   @out_mesh:           mesh without noise points
def delete_noise(mesh, threshold=2000):
    tooth_split = trimesh.graph.split(mesh, only_watertight=False)
    # only get the teeth whose connected points number is larger than the threshold 
    out_tooth =  [k for k in tooth_split if k.vertices.shape[0] > threshold]
    out_mesh = sum(out_tooth)
    return out_mesh


# refine_mesh:              split the teeth until each connected component's point is less than threshold
# input:
#   @mesh:                  a trimesh type mesh
#   @threshold:             if the connected component's point number is larger than the thresshold, it will be splitted
#   @level:                 the level of neighbors that will be used to get the average normal angel. 
#   @removed_threshold:     the percentage of points that will be used to remove surfaces.
# return:
#   @good_splitted_teeth:   refined mesh 
def refine_mesh(mesh, threshold,level, removed_threshold):                 
    tooth_splitted = trimesh.graph.split(mesh,only_watertight=False)
    # get the teeth whose connected points number is larger than the threshold 
    bad_splitted_teeth =  [k for k in tooth_splitted if k.vertices.shape[0] > threshold]
    good_splitted_teeth = sum([k for k in tooth_splitted if k not in bad_splitted_teeth])
    
    # until each connected component's point is less than threshold
    while not len(bad_splitted_teeth) == 0:
        for tooth in bad_splitted_teeth:
            mask = get_removed_mask(tooth, level, removed_threshold)
            tooth.update_faces(mask)
        new_teeth = sum(bad_splitted_teeth)
        sub_splitted_teeth = trimesh.graph.split(new_teeth,only_watertight=False)
        bad_splitted_teeth =  [k for k in sub_splitted_teeth if k.vertices.shape[0] > threshold]
        good_sub_splitted_teeth = sum([k for k in sub_splitted_teeth if k not in bad_splitted_teeth])
        good_splitted_teeth += good_sub_splitted_teeth
    return good_splitted_teeth


# get_neighbors: get the n level neighbors of the point based on the point index
# input:
#   @neighbors_array:    a 2D array [n,neighbors_n]
#   @point_idx:          int, the point index
#   @level:              the level of neighbors that will be used to get the average normal angel.
# return:
#   @neighbors:          a 1D array [neighbors_n], elements are the index of neighbors
def get_neighbors(neighbors_array, point_idx, level):
    neighbors = np.asarray(neighbors_array[point_idx])

    # get the neighbors based on the level we set
    for l in range(level-1):
        neighbors = np.asarray(list(set(list(chain.from_iterable(neighbors_array[neighbors])))))
    return neighbors


# get_removed_mask: get the mask for removing high curvature faces. 
# input:
#   @mesh:            input mesh
#   @level:           the level of neighbors that will be used to get the average normal angel.
#   @threshold:      the percentage of points that will be used to remove surfaces.
# return:
#   @mask:            a 1D list, elements are the index of faces that should be removed 
def get_removed_mask(mesh, level = 2, threshold=0.2):
    

    faces = mesh.faces
    points = mesh.vertices

    # get the normal of each points
    normals = mesh.vertex_normals

    # get the neighbors of each vertices
    neighbors_array = np.asarray(mesh.vertex_neighbors)

    # set the average normal angle list
    average_angle = []

    # iterate each point, get its N level neighbors, calculate the average normal angles. 
    for point_idx in tqdm(range(len(points))):
        neighbors = get_neighbors(neighbors_array, point_idx, level)
        # calcuate the angles
        neighbors_normals = normals[neighbors,:]
        cosine_array = np.dot(normals[point_idx].reshape(1,-1),neighbors_normals.T)
        angle_array = np.nan_to_num(np.arccos(cosine_array))
        average_angle.append(np.mean(angle_array))

    
    # the number of points that will be used to remove surfaces.
    removed_n = int(len(average_angle)*threshold)

    # the index of points that will be used to remove surfaces.
    removed_point = np.argpartition(average_angle, -removed_n)[-removed_n:]
    # removed_point = np.where(average_angle>0.8*np.max(average_angle))[0]
    removed_point_mask = set(removed_point)

    ################### if want to output the curvature results ###################
    # export_cur2color('color.txt',average_angle,points, removed_point)
    # export_cur('cur3.txt', average_angle, points)
    ############################################################################

    # remove the faces that contains the points have high curvature. 
    mask = []   # the index of faces that should be removed 
    for i in range(len(faces)):
        removed_points = removed_point_mask & set(faces[i])
        if not len(removed_points) == 0:
            continue
        mask.append(i)
    return mask

if __name__=="__main__":


    # load input mesh, get faces and points
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_path', help='Input mesh path')
    mesh = trimesh.load(parser.parse_args().mesh_path)

    mesh = delete_noise(mesh, threshold=2000)

    old_vertices = mesh.vertices

    # get the mask
    mask = get_removed_mask(mesh, level = 2, threshold=0.2)

    # remove the faces
    mesh.update_faces(mask)

    # refine the mesh and get the removed faces' mask
    mesh = refine_mesh(mesh, threshold= 20000,level = 2, removed_threshold=0.20)
    
    new_vertices = mesh.vertices

    # get the deleted mesh
    deleted_points = get_deleted_points(old_vertices, new_vertices)
    deleted_points.export('deleted.obj')
    
    # export the mesh
    mesh.export('text.stl')

    # TODO: use trimesh.graph.split(mesh, only_watertight=False)
    # to get the connected components 

    # TODO: calculte each components' centroid

    # TODO: connect the points to its nearest conponents
