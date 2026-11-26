import trimesh
import argparse
import numpy as np
import os
from typing import DefaultDict
from collections import defaultdict
from tqdm import tqdm
from trimesh.exchange.load import mesh_formats
from trimesh.graph import neighbors
import scipy
import logging as log
import pickle
import heapq
from itertools import chain
import open3d as o3d
from typing import DefaultDict
import argparse

parser = argparse.ArgumentParser(description = 'Merge_Fusion_step')
parser.add_argument('-tooth_scan_dir', '--tooth_scan_dir', default = '', help='tooth_scan_dir')
parser.add_argument('-tooth_cbct_origin', '--tooth_cbct_origin', default = '', help='tooth_cbct_origin')
parser.add_argument('-ismirror', '--ismirror', default = '', help='ismirror')
args = parser.parse_args()


teethdir = args.tooth_scan_dir

def fun_deleted_points(old_vertices, new_vertices):
    deleted_points = []
    new_vertices = trimesh.points.PointCloud(new_vertices)
    dict = DefaultDict(int)
    # import pdb; pdb.set_trace()
    for vertice in new_vertices:
        dict[str(vertice)] = len(dict)
    for vertice in old_vertices:
        if not str(vertice) in dict:
            deleted_points.append(vertice)
    points = trimesh.points.PointCloud(deleted_points)
    return points

def fun_deleted_points_1(pcd_target, pcd_target_close):
    deleted_points = []
    dict = DefaultDict(int)
    num_dic = []
    pcd_target_1 = np.asarray(pcd_target.points)
    pcd_target_close_1 = np.asarray(pcd_target_close.points)

    for vertice in pcd_target_close_1:
        dict[str(vertice)] = len(dict)
    for i, vertice in enumerate(pcd_target_1):
        if not str(vertice) in dict:
            deleted_points.append(vertice)
            num_dic.append(i)

    pcd_target_close_normals = np.asarray(pcd_target.normals)
    new_normals = pcd_target_close_normals[num_dic]


    points = trimesh.points.PointCloud(deleted_points)

    a = np.asarray(points.vertices)
    point_o3d = o3d.utility.Vector3dVector(a)
    point_o3d = o3d.geometry.PointCloud(point_o3d)

    b = new_normals
    point_o3d.normals = o3d.utility.Vector3dVector(b)

    return point_o3d

def get_normals(mesh , pcd):

    vertices_normal = sorted(zip(mesh.vertices, mesh.vertex_normals), key=lambda x: (x[0][0], x[0][1], x[0][2]))
    vertices, normals = zip(*vertices_normal)

    # sort the vertices of point cloud
    normals = np.asarray(normals)
    points = np.asarray(pcd.points)
    points = sorted(points, key=lambda x: (x[0], x[1], x[2]))

    # delete duplicated points in point cloud
    delete_list = []
    for i in range(1, len(points)):
        if (points[i] == points[i - 1]).all():
            delete_list.append(i)
    points = np.delete(points, delete_list, axis=0)

    # map the points of point cloud to vertices of mesh
    i, j = 0, 0
    normals_idx = []
    pre_j = 0
    while i < len(points):
        while not (vertices[j] == points[i]).all():
            j += 1
        if pre_j == j and j != 0:
            print(points[i], points[i - 1])
            print(normals[i], normals[i - 1])
            print("error occured")
            exit(0)
        pre_j = j
        normals_idx.append(j)
        i += 1
        j += 1
        if i % 100000 == 0:
            print("{}/{}".format(i, len(points)))

    # get the normals
    normals_out = normals[normals_idx]

    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(points)
    pcd_out.normals = o3d.utility.Vector3dVector(normals_out)
    return pcd_out

def point_merge(merge_step, deleted_points, teeth_mesh_list):
    dict = defaultdict()
    deleted_points = deleted_points.vertices
    per_merge_num = deleted_points.shape[0] // merge_step
    last_merge_num = deleted_points.shape[0] - (merge_step - 1) * per_merge_num

    # convert mesh to point clouds
    teeth_pcd_list = []
    for i in range(len(teeth_mesh_list)):
        teeth_pcd_list.append(trimesh.points.PointCloud(teeth_mesh_list[i].vertices))

    for step in range(merge_step):
        log.info("point_merge: {}/{}".format(step + 1, merge_step))
        distance_dic = []
        if step == merge_step - 1:
            per_merge_num = last_merge_num

        for i in tqdm(range(len(teeth_mesh_list))):
            # create a tree for current tooth
            tree = scipy.spatial.cKDTree(teeth_mesh_list[i].vertices)
            # get the distance of deleted points to the current tooth
            distance_dic.append(np.mean(tree.query(deleted_points, 3)[0], axis=1))

        # cla the closed points to mehsh,and merge

        distance_dic = np.asarray(distance_dic)
        points_min_dist = np.min(distance_dic, axis=0)

        merge_points_list = heapq.nsmallest(per_merge_num, range(len(points_min_dist)), points_min_dist.__getitem__)
        points_labels = np.argmin(distance_dic, axis=0)

        for i in range(len(teeth_mesh_list)):
            points = np.where(points_labels == i)[0]
            points = list(set(points) & set(merge_points_list))
            vertices = deleted_points[points]
            if i not in dict:
                dict[i] = trimesh.points.PointCloud(vertices)
            else:
                dict[i] += trimesh.points.PointCloud(vertices)
            teeth_mesh_list[i] += trimesh.points.PointCloud(vertices)
            #import pdb;pdb.set_trace()
            #teeth_mesh_list[i].export('code/PointSeg/reconstruct/reconstruct_stage_%d.ply' % i)
        mylist = [dict[i] for i in range(len(teeth_mesh_list))]
        deleted_points = np.delete(deleted_points, merge_points_list, axis=0)

    return mylist


def get_neighbors(neighbors_array, point_idx, level):
    neighbors = np.asarray(neighbors_array[point_idx])

    # get the neighbors based on the level we set
    for l in range(level - 1):
        neighbors = np.asarray(list(set(list(chain.from_iterable(neighbors_array[neighbors])))))
    return neighbors

def get_removed_mask(mesh, level = 2, threshold = 0.15 ):                 #改为points
    faces = mesh.faces
    points = mesh.vertices

    # get the normal of each points
    normals = mesh.vertex_normals

    # get the neighbors of each vertices
    neighbors_array = np.asarray(mesh.vertex_neighbors)

    # set the average normal angle list
    average_angle = []

    # iterate each point, get its N level neighbors, calculate the average normal angles.
    for point_idx in range(len(points)):
        neighbors = get_neighbors(neighbors_array, point_idx, level)
        # calcuate the angles
        neighbors_normals = normals[neighbors, :]
        cosine_array = np.dot(normals[point_idx].reshape(1, -1), neighbors_normals.T)
        angle_array = np.nan_to_num(np.arccos(cosine_array))
        average_angle.append(np.mean(angle_array))

    # the number of points that will be used to remove surfaces.
    removed_n = int(len(average_angle) * threshold)

    # the index of points that will be used to remove surfaces.
    removed_point = np.argpartition(average_angle, -removed_n)[-removed_n:]
    # removed_point = np.where(average_angle>0.8*np.max(average_angle))[0]
    removed_point_mask = set(removed_point)

    mask = []  # the index of faces that should be removed
    for i in range(len(faces)):
        removed_points = removed_point_mask & set(faces[i])
        if not len(removed_points) == 0:
            continue
        mask.append(i)
    return mask


def fusion_stage(mesh, points,teeth_scan):
    temp_teeth = args.tooth_scan_dir
    strlist = temp_teeth.split('teeth_seg/')
    temp = args.ismirror
    temp_str = (strlist[1])[0]
    temp_mark = (strlist[1])[0]
    if temp == '1' and temp_str == 'L':
        temp_mark = 'U'
    if temp == '1' and temp_str == 'U':
        temp_mark = 'L'
    # import pdb;pdb.set_trace()
    if temp_mark == 'L' :
        mesh_cent = mesh

        mesh_points_cent = np.where(np.asarray(mesh.vertices)[:, 2] > mesh_cent.centroid[2])
        vertices_z_cent = np.asarray(mesh.vertices)[mesh_points_cent]
        vertices_z_normals_cent = np.asarray(mesh.vertex_normals)[mesh_points_cent]
        vertices_z_cent = trimesh.points.PointCloud(vertices_z_cent)
        vertices_z_normals_cent = trimesh.points.PointCloud(vertices_z_normals_cent)

        a1 = np.asarray(vertices_z_cent.vertices)
        vertices_z_cent = o3d.utility.Vector3dVector(a1)
        vertices_z_cent = o3d.geometry.PointCloud(vertices_z_cent)

        b1 = np.asarray(vertices_z_normals_cent.vertices)
        vertices_z_cent.normals = o3d.utility.Vector3dVector(b1)

        pcd_target_close = cal_distance(teeth_scan, vertices_z_cent)
        # import pdb;pdb.set_trace()
        vertices_z_cent = fun_deleted_points_1(vertices_z_cent, pcd_target_close)                                               #到底是计算距离近的好，还是

        common_points = vertices_z_cent
        points_cent_left = np.where(np.asarray(points.points)[:, 2] < mesh_cent.centroid[2])

        points_z_cent_left = np.asarray(points.points)[points_cent_left]
        points_z_normals_cent_left = np.asarray(points.normals)[points_cent_left]
        new_points_z_cent_left = trimesh.points.PointCloud(points_z_cent_left)
        new_points_z_normals_cent_left = trimesh.points.PointCloud(points_z_normals_cent_left)

        a3 = np.asarray(new_points_z_cent_left.vertices)
        point_left = o3d.utility.Vector3dVector(a3)
        point_left = o3d.geometry.PointCloud(point_left)

        b3 = np.asarray(new_points_z_normals_cent_left.vertices)
        point_left.normals = o3d.utility.Vector3dVector(b3)

        point_all = point_left + common_points
    else:
        mesh_cent = mesh

        mesh_points_cent = np.where(np.asarray(mesh.vertices)[:, 2] < mesh_cent.centroid[2])
        vertices_z_cent = np.asarray(mesh.vertices)[mesh_points_cent]
        vertices_z_normals_cent = np.asarray(mesh.vertex_normals)[mesh_points_cent]
        vertices_z_cent = trimesh.points.PointCloud(vertices_z_cent)
        vertices_z_normals_cent = trimesh.points.PointCloud(vertices_z_normals_cent)

        a1 = np.asarray(vertices_z_cent.vertices)
        vertices_z_cent = o3d.utility.Vector3dVector(a1)
        vertices_z_cent = o3d.geometry.PointCloud(vertices_z_cent)

        b1 = np.asarray(vertices_z_normals_cent.vertices)
        vertices_z_cent.normals = o3d.utility.Vector3dVector(b1)

        pcd_target_close = cal_distance(teeth_scan, vertices_z_cent)
        # import pdb;pdb.set_trace()
        vertices_z_cent = fun_deleted_points_1(vertices_z_cent, pcd_target_close)
        
        common_points = vertices_z_cent
        points_cent_left = np.where(np.asarray(points.points)[:, 2] > mesh_cent.centroid[2])

        points_z_cent_left = np.asarray(points.points)[points_cent_left]
        points_z_normals_cent_left = np.asarray(points.normals)[points_cent_left]
        new_points_z_cent_left = trimesh.points.PointCloud(points_z_cent_left)
        new_points_z_normals_cent_left = trimesh.points.PointCloud(points_z_normals_cent_left)

        a3 = np.asarray(new_points_z_cent_left.vertices)
        point_left = o3d.utility.Vector3dVector(a3)
        point_left = o3d.geometry.PointCloud(point_left)

        b3 = np.asarray(new_points_z_normals_cent_left.vertices)
        point_left.normals = o3d.utility.Vector3dVector(b3)

        point_all = point_left + common_points
    return point_all

def cal_distance(pcd_source,pcd_target):
    distance_dic = []
    tree = scipy.spatial.cKDTree(np.asarray(pcd_source.points))
    temp_teeth = args.tooth_scan_dir
    strlist = temp_teeth.split('teeth_seg/')
    if (strlist[1])[0] == 'L':
        points = np.where(np.asarray(pcd_target.points)[:, 2] > float(np.min(np.asarray(pcd_source.points)[:, 2])))
        vertices_z = np.asarray(pcd_target.points)[points]
        vertices_z_normals = np.asarray(pcd_target.normals)[points]

        new_points_z = trimesh.points.PointCloud(vertices_z)
        new_points_z_normals = trimesh.points.PointCloud(vertices_z_normals)

        distance_dic.append(np.mean(tree.query(new_points_z.vertices, 3)[0], axis=1))

        # cla the closed points to mehsh,and merge
        distance_dic = np.asarray(distance_dic)

        distance_dic = distance_dic[0]
        #top_idx = distance_dic.argsort()[::-1][0:int(distance_dic.shape[0] * 0.2)]
        top_idx = distance_dic.argsort()[::][0:int(distance_dic.shape[0] * 0.2)]        #设置Z轴，然后再选30%的点    origin:0.6

        vertices = new_points_z.vertices[top_idx]
        normals = new_points_z_normals.vertices[top_idx]

        new_points = trimesh.points.PointCloud(vertices)
        a = np.asarray(new_points.vertices)
        point_o3d = o3d.utility.Vector3dVector(a)
        point_o3d = o3d.geometry.PointCloud(point_o3d)

        new_points_normals = trimesh.points.PointCloud(normals)
        b = np.asarray(new_points_normals.vertices)
        point_o3d.normals = o3d.utility.Vector3dVector(b)
    else:
        points = np.where(np.asarray(pcd_target.points)[:, 2] < float(np.max(np.asarray(pcd_source.points)[:, 2])))
        vertices_z = np.asarray(pcd_target.points)[points]
        vertices_z_normals = np.asarray(pcd_target.normals)[points]

        new_points_z = trimesh.points.PointCloud(vertices_z)
        new_points_z_normals = trimesh.points.PointCloud(vertices_z_normals)

        distance_dic.append(np.mean(tree.query(new_points_z.vertices, 3)[0], axis=1))

        # cla the closed points to mehsh,and merge
        distance_dic = np.asarray(distance_dic)

        distance_dic = distance_dic[0]
        # top_idx = distance_dic.argsort()[::-1][0:int(distance_dic.shape[0] * 0.2)]
        top_idx = distance_dic.argsort()[::][0:int(distance_dic.shape[0] * 0.2)]  # 设置Z轴，然后再选30%的点    origin:0.6

        vertices = new_points_z.vertices[top_idx]
        normals = new_points_z_normals.vertices[top_idx]

        new_points = trimesh.points.PointCloud(vertices)
        a = np.asarray(new_points.vertices)
        point_o3d = o3d.utility.Vector3dVector(a)
        point_o3d = o3d.geometry.PointCloud(point_o3d)

        new_points_normals = trimesh.points.PointCloud(normals)
        b = np.asarray(new_points_normals.vertices)
        point_o3d.normals = o3d.utility.Vector3dVector(b)

    return point_o3d

if __name__ == "__main__":
    point_left = trimesh.load(r'code/PointSeg/pcd_1_new.ply')
    mesh = trimesh.load(args.tooth_cbct_origin)
    teethfiles = os.listdir(teethdir)
    teethdic = []
    filename=[]
    # read mesh as array
    for name in teethfiles:
        file = os.path.join(teethdir, name)
        teeth = trimesh.load(file)
        teethdic.append(teeth)
        filename.append(name)
        # import pdb;pdb.set_trace()

    distance_dic = []
    # turn to array
    point_array = np.asarray(point_left.vertices)
    #####################cal the ditance of point and mesh
    mesh_to_point = []
    for i in tqdm(range(len(teethdic))):
        mesh_to_point.append(trimesh.points.PointCloud(teethdic[i].vertices))

    teeth_pcd_list = point_merge(20 ,point_left, mesh_to_point)

    for i in tqdm(range(len(teethdic))):
        teeth_scan = np.asarray(teethdic[i].vertices)
        teeth_scan_normals = np.asarray(teethdic[i].vertex_normals)
        teeth_scan = o3d.utility.Vector3dVector(teeth_scan)
        teeth_scan = o3d.geometry.PointCloud(teeth_scan)
        teeth_scan.normals = o3d.utility.Vector3dVector(teeth_scan_normals)
        # if args.ismirror == '1':
        #     import pdb;pdb.set_trace()
        #     teeth_scan.normals = o3d.utility.Vector3dVector(-np.asarray(teeth_scan.normals))
        #teeth_pcd_list[i]  = fun_deleted_points(teeth_pcd_list[i],teethdic[i].vertices)
        a = np.asarray(teeth_pcd_list[i].vertices)
        teeth_part = o3d.utility.Vector3dVector(a)
        teeth_part = o3d.geometry.PointCloud(teeth_part)
        # import pdb;pdb.set_trace()
        #o3d.io.write_point_cloud('code/PointSeg/reconstruct/reconstruct_points_stage_%d.ply' % i, points)
        #teeth_pcd_list[i].export('code/PointSeg/reconstruct/%d.ply' % i)
        try:
            teeth_part = get_normals(mesh, teeth_part)
        except:
            continue

        teeth_all_stage = teeth_part
        mesh1, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(teeth_all_stage, depth = 7,  width=0, scale=1.1, linear_fit=False, n_threads=64)
        #o3d.io.write_point_cloud("copy_of_fragment.ply", pcd_1)
        o3d.io.write_triangle_mesh('code/PointSeg/reconstruct_stage_%d.ply'%i , mesh1)
        mesh1 = trimesh.load('code/PointSeg/reconstruct_stage_%d.ply'%i)
        mask = get_removed_mask(mesh1)
        mesh1.update_faces(mask)
        tooth_split = trimesh.graph.split(mesh1, only_watertight=False)
        out_tooth = [k for k in tooth_split if k.vertices.shape[0] > 500]              #可能犯错
        mesh1 = out_tooth[0]
        for num in range(len(out_tooth)):
            mesh1 += out_tooth[num]
        #####此处
        #mesh1.export('code/PointSeg/reconstruct/mesh1_%d.ply'%i)
        points = fusion_stage(mesh1, teeth_part, teeth_scan)                 #mesh1:trimesh      points:open3d

        o3d.io.write_point_cloud('code/PointSeg/reconstruct/points_%d.ply' % i, points)

        # pcd_target_close = cal_distance(teeth_scan, points)
        # # import pdb;pdb.set_trace()
        # points = fun_deleted_points_1(points, pcd_target_close)                                               #到底是计算距离近的好，还是


        o3d.io.write_point_cloud('code/PointSeg/reconstruct/reconstruct_points_stage_%d.ply' % i, points)
        points = points + teeth_scan
        mesh1, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(points, depth = 8,  width=0, scale=1.1, linear_fit=False, n_threads=64)
        # import pdb;pdb.set_trace()
        o3d.io.write_triangle_mesh('code/PointSeg/reconstruct/reconstruct_stage_%d.ply'%i, mesh1)
        # import pdb;pdb.set_trace()
        mesh1 = trimesh.load('code/PointSeg/reconstruct/reconstruct_stage_%d.ply'%i)
        #mesh1 = mesh1 + teethdic[i]
        mesh1 = trimesh.smoothing.filter_laplacian(mesh1, iterations = 8)
        temp_teeth = args.tooth_scan_dir
        strlist = temp_teeth.split('teeth_seg/')

        tooth_split = trimesh.graph.split(mesh1, only_watertight=False)
        out_tooth = [k for k in tooth_split if k.vertices.shape[0] > 500]              #删除可能出现的碎片
        mesh1 = out_tooth[0]
        for num in range(len(out_tooth)):
            mesh1 += out_tooth[num]

        filename_basename=os.path.basename(filename[i])
        mesh1.export(strlist[0]+'/fusion/'+(strlist[1])[0]+'_reconstruct_%s.ply'%filename_basename)














