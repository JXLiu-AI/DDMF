import open3d as o3d
import numpy as np
import trimesh
import tqdm
import scipy
from typing import DefaultDict
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser(description = 'multi-modal fusion')
parser.add_argument('-tooth_scan_match', '--tooth_scan_match', default = '', help='tooth_scan_match')
parser.add_argument('-tooth_cbct_match', '--tooth_cbct_match', default = '', help='tooth_cbct_match')
parser.add_argument('-tooth_cbct_origin', '--tooth_cbct_origin', default = '', help='tooth_cbct_origin')
parser.add_argument('-ismirror', '--ismirror', default = '', help='ismirror')
parser.add_argument('-tooth_scan_dir', '--tooth_scan_dir', default = '', help='tooth_scan_dir')
args = parser.parse_args()

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

def fun_deleted_points(pcd_target, pcd_target_close):
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

def cal_distance(pcd_source,pcd_target):
    distance_dic = []
    distance_dic_cent = []
    tree = scipy.spatial.cKDTree(np.asarray(pcd_source.points))
    pcd_source_1 = trimesh.load(args.tooth_scan_match)
    #######################################
    temp_teeth = args.tooth_scan_dir
    strlist = temp_teeth.split('teeth_seg/')
    temp = args.ismirror
    # import pdb;pdb.set_trace()
    temp_str = (strlist[1])[0]
    temp_mark = (strlist[1])[0]
    # import pdb;pdb.set_trace()
    if temp == '1' and temp_str == 'L':
        temp_mark = 'U'
    if temp == '1' and temp_str == 'U':
        temp_mark = 'L'
    # import pdb;pdb.set_trace()
    if temp_mark == 'L':
    ######################################
        points = np.where(np.asarray(pcd_target.points)[:, 2] > float(np.min(np.asarray(pcd_source.points)[:, 2])))
        points_cent = np.where(np.asarray(pcd_target.points)[:, 2] > (float(np.min(np.asarray(pcd_source.points)[:, 2]))+ pcd_source_1.centroid[2])/2)

        vertices_z = np.asarray(pcd_target.points)[points]
        vertices_z_normals = np.asarray(pcd_target.normals)[points]
        vertices_z_cent = np.asarray(pcd_target.points)[points_cent]
        vertices_z_normals_cent = np.asarray(pcd_target.normals)[points_cent]

        new_points_z = trimesh.points.PointCloud(vertices_z)
        new_points_z_normals = trimesh.points.PointCloud(vertices_z_normals)

        new_points_z_cent = trimesh.points.PointCloud(vertices_z_cent)
        new_points_z_normals_cent = trimesh.points.PointCloud(vertices_z_normals_cent)


        distance_dic.append(np.mean(tree.query(new_points_z.vertices, 3)[0], axis=1))
        distance_dic_cent.append(np.mean(tree.query(new_points_z_cent.vertices, 3)[0], axis=1))
        # cla the closed points to mehsh,and merge
        distance_dic = np.asarray(distance_dic)
        distance_dic_cent = np.asarray(distance_dic_cent)

        distance_dic = distance_dic[0]
        distance_dic_cent = distance_dic_cent[0]
        #top_idx = distance_dic.argsort()[::-1][0:int(distance_dic.shape[0] * 0.2)]
        top_idx = distance_dic.argsort()[::][0:int(distance_dic.shape[0] * 0.3)]        #至少0.2！设置Z轴，然后再选30%的点  0.3
        top_idx_cent = distance_dic_cent.argsort()[::][0:int(distance_dic_cent.shape[0] * 0)]  # 设置Z轴，然后再选50%的点    不选了

        vertices = new_points_z.vertices[top_idx]
        normals = new_points_z_normals.vertices[top_idx]
        vertices_cent = new_points_z_cent.vertices[top_idx_cent]
        normals_cent = new_points_z_normals_cent.vertices[top_idx_cent]

        new_points = trimesh.points.PointCloud(vertices)
        new_points_cent = trimesh.points.PointCloud(vertices_cent)
        new_points += new_points_cent
        a = np.asarray(new_points.vertices)
        point_o3d = o3d.utility.Vector3dVector(a)
        point_o3d = o3d.geometry.PointCloud(point_o3d)

        new_points_normals = trimesh.points.PointCloud(normals)
        new_points_normals_cent = trimesh.points.PointCloud(normals_cent)
        new_points_normals += new_points_normals_cent
        b = np.asarray(new_points_normals.vertices)
        point_o3d.normals = o3d.utility.Vector3dVector(b)
    if temp_mark == 'U':

        points = np.where(np.asarray(pcd_target.points)[:, 2] < float(np.max(np.asarray(pcd_source.points)[:, 2])))
        points_cent = np.where(np.asarray(pcd_target.points)[:, 2] < (float(np.max(np.asarray(pcd_source.points)[:, 2])) + pcd_source_1.centroid[2]) / 2)

        vertices_z = np.asarray(pcd_target.points)[points]
        vertices_z_normals = np.asarray(pcd_target.normals)[points]
        vertices_z_cent = np.asarray(pcd_target.points)[points_cent]
        vertices_z_normals_cent = np.asarray(pcd_target.normals)[points_cent]

        new_points_z = trimesh.points.PointCloud(vertices_z)
        new_points_z_normals = trimesh.points.PointCloud(vertices_z_normals)

        new_points_z_cent = trimesh.points.PointCloud(vertices_z_cent)
        new_points_z_normals_cent = trimesh.points.PointCloud(vertices_z_normals_cent)

        distance_dic.append(np.mean(tree.query(new_points_z.vertices, 3)[0], axis=1))
        distance_dic_cent.append(np.mean(tree.query(new_points_z_cent.vertices, 3)[0], axis=1))
        # cla the closed points to mehsh,and merge
        distance_dic = np.asarray(distance_dic)
        distance_dic_cent = np.asarray(distance_dic_cent)

        distance_dic = distance_dic[0]
        distance_dic_cent = distance_dic_cent[0]
        # top_idx = distance_dic.argsort()[::-1][0:int(distance_dic.shape[0] * 0.2)]
        top_idx = distance_dic.argsort()[::][0:int(distance_dic.shape[0] * 0.3)]  # 至少0.2！设置Z轴，然后再选30%的点  0.3
        top_idx_cent = distance_dic_cent.argsort()[::][0:int(distance_dic_cent.shape[0] * 0)]  # 设置Z轴，然后再选50%的点    不选了

        vertices = new_points_z.vertices[top_idx]
        normals = new_points_z_normals.vertices[top_idx]
        vertices_cent = new_points_z_cent.vertices[top_idx_cent]
        normals_cent = new_points_z_normals_cent.vertices[top_idx_cent]

        new_points = trimesh.points.PointCloud(vertices)
        new_points_cent = trimesh.points.PointCloud(vertices_cent)
        new_points += new_points_cent
        a = np.asarray(new_points.vertices)
        point_o3d = o3d.utility.Vector3dVector(a)
        point_o3d = o3d.geometry.PointCloud(point_o3d)

        new_points_normals = trimesh.points.PointCloud(normals)
        new_points_normals_cent = trimesh.points.PointCloud(normals_cent)
        new_points_normals += new_points_normals_cent
        b = np.asarray(new_points_normals.vertices)
        point_o3d.normals = o3d.utility.Vector3dVector(b)
    return point_o3d

def delete_tiny(pcd_1):
    distances = pcd_1.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        cluster = pcd_1.cluster_dbscan(radius, 10, print_progress=True)
        labels = np.array(cluster)

    max_label = max(cluster)

    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))                      # 可视化
    colors[labels < 0] = 0
    pcd_1.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.io.write_point_cloud("code/PointSeg/copy_of_fragment.ply", pcd_1)                                           #可视化聚类结果
    lens = dict()
    mark = []
    for i in range(max_label + 1):
        lens[i] = np.sum(np.asarray(cluster) == i)
        if lens[i] < 5000:
            mark.append(i)

    pcd_1_tiny = []
    pcd_1_tiny_normals = []
    for i in range(len(mark)):
        idx = np.where(labels == mark[i])
        pcd_1_array = np.asarray(pcd_1.points)
        pcd_1_array_normals = np.asarray(pcd_1.normals)
        pcd_1_tiny.append(pcd_1_array[idx])
        pcd_1_tiny_normals.append(pcd_1_array_normals[idx])
    # import pdb;pdb.set_trace()
    if pcd_1_tiny == []:
        pcd_1_new = pcd_1
    else:
        pcd_1_tiny_new = pcd_1_tiny[0]
        pcd_1_tiny_new_normals = pcd_1_tiny_normals[0]
        for i in range(len(pcd_1_tiny) - 1):
            pcd_1_tiny_new = np.concatenate([pcd_1_tiny_new, pcd_1_tiny[i + 1]], axis=0)
            pcd_1_tiny_new_normals = np.concatenate([pcd_1_tiny_new_normals, pcd_1_tiny_normals[i + 1]], axis=0)

        point_1_tiny = o3d.utility.Vector3dVector(pcd_1_tiny_new)
        point_1_tiny = o3d.geometry.PointCloud(point_1_tiny)
        point_1_tiny.normals = o3d.utility.Vector3dVector(pcd_1_tiny_new_normals)

        pcd_1_new = fun_deleted_points(pcd_1, point_1_tiny)
    return pcd_1_new


if __name__ == '__main__':
    pcd_source = o3d.io.read_point_cloud(args.tooth_scan_match)
    pcd_cbct_origin = trimesh.load(args.tooth_cbct_origin)
    pcd_target_no_normals = o3d.io.read_point_cloud(args.tooth_cbct_match)
    pcd_target = get_normals(pcd_cbct_origin, pcd_target_no_normals)
    if args.ismirror == '1':
        # import pdb;pdb.set_trace()
        pcd_source.normals = o3d.utility.Vector3dVector(-np.asarray(pcd_source.normals))
    else:
        pcd_source.normals = o3d.utility.Vector3dVector(np.asarray(pcd_source.normals))

    pcd_target_close = cal_distance(pcd_source,pcd_target)
    pcd_target_new = fun_deleted_points(pcd_target,pcd_target_close)

    pcd_1, ind = pcd_target_new.remove_statistical_outlier(nb_neighbors = 200, std_ratio = 1)    #删除离群点,need open3d origin:100
    # pcd_1, ind = pcd_target_new.remove_radius_outlier(nb_points=10, radius=1)                #删除离群点
    ################################
    pcd_1_new = delete_tiny(pcd_1)
    #################################
    o3d.io.write_point_cloud('code/PointSeg/pcd_1_new.ply',pcd_1_new)
    pcd_fusion = pcd_1_new + pcd_source
    #o3d.io.write_point_cloud('pcd_1_new.ply', pcd_1_new)
    o3d.io.write_point_cloud('code/PointSeg/Fusion3.ply', pcd_fusion)

    print('run Poisson surface reconstruction')
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_fusion, depth = 9,  width=0, scale=1.1, linear_fit=False, n_threads=64)              #origin: 9
    o3d.io.write_triangle_mesh('code/PointSeg/reconstruct_o3d.ply', mesh)
    mesh = trimesh.load('code/PointSeg/reconstruct_o3d.ply')
    mesh = trimesh.smoothing.filter_laplacian(mesh, iterations = 6)                                                    #possion reconstruction

    temp = args.tooth_scan_dir
    # import pdb;pdb.set_trace()
    strlist = temp.split('teeth_seg/')
    mesh.export(strlist[0]+'/fusion/'+(strlist[1])[0]+'_reconstruct_all.ply')
    os.system('python code/PointSeg/Merge_Fusion_step_3.py --tooth_scan_dir '+args.tooth_scan_dir+\
              ' --tooth_cbct_origin '+args.tooth_cbct_origin+' --ismirror '+ str(args.ismirror))






