import open3d as o3d
import numpy as np
import copy
import trimesh
import os
import argparse

parser = argparse.ArgumentParser(description = 'Global registration')
parser.add_argument('-tooth_scan_dir', '--tooth_scan_dir', default = '', help='tooth_scan_dir')
parser.add_argument('-tooth_cbct', '--tooth_cbct', default = '', help='tooth_cbct')
parser.add_argument('-tooth_cbct_origin', '--tooth_cbct_origin', default = '', help='tooth_cbct_origin')
parser.add_argument('-hardsample', '--hard', type = int ,default = 0, help = 'hardsample(points less)')
parser.add_argument('--output_path', type=str, required=True)
args = parser.parse_args()
os.makedirs(args.output_path, exist_ok=True)
tooth_scan_path = os.path.join(args.output_path, 'tooth_alltooth.ply')

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

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(cbct , voxel_size,  ismirror):
    print(":: Load two point clouds and disturb initial pose.")
    source_1 = o3d.io.read_point_cloud(tooth_scan_path)
    mesh_scan = o3d.io.read_triangle_mesh(tooth_scan_path)
    source_1.normals = mesh_scan.vertex_normals
    if ismirror == 1:
        trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],  # transform matrix
                                 [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    else:
        trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],  # 
                                 [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    if cbct == 'U':
        cbct_name = args.tooth_cbct + '/U_all.ply'

        target = o3d.io.read_point_cloud(cbct_name)
        target_origin = trimesh.load(args.tooth_cbct_origin)
        target = get_normals(target_origin, target)
    else:
        target = o3d.io.read_point_cloud(args.tooth_cbct + '/L_all.ply')
        target_origin = trimesh.load(args.tooth_cbct_origin)
        target = get_normals(target_origin, target)
    # if args.hard == 1:
    #     print('hardsample')
    # else:
    #     target = target.voxel_down_sample(1)  # 

    # if ismirror == 1:
    #     #import pdb;pdb.set_trace()
    #     source_1.normals = o3d.utility.Vector3dVector(-np.asarray(source_1.normals))
    source_1.transform(trans_init)


    source_1_down, source_1_fpfh = preprocess_point_cloud(source_1, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source_1, target, source_1_down, target_down, source_1_fpfh, target_fpfh


def execute_global_registration(source_1_down, target_down, source_1_fpfh,
                                target_fpfh, voxel_size, ismirror):
    distance_threshold = voxel_size * 1   #1.5 #2
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    # import pdb;pdb.set_trace()
    # if ismirror == 1:
    #     #source_1_down
    #     source_1_down.normals = o3d.utility.Vector3dVector(-np.asarray(source_1_down.normals))

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_1_down, target_down, source_1_fpfh, target_fpfh,True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),           #origin
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(0.9)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(1000000000, 0.7))  
    return result


def refine_registration(source_1, target, source_1_fpfh, target_fpfh, voxel_size,result_ransac):
    distance_threshold = voxel_size * 0.6      #0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source_1, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    result_1 = o3d.pipelines.registration.registration_icp(
        source_1, target, distance_threshold*1/2, result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    result_2 = o3d.pipelines.registration.registration_icp(
        source_1, target, distance_threshold*1/3,  result_1.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    return result_2


def init( cbct ,ismirror):
    #global trans_init
    # if ismirror == 1:
    #     trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],  
    #                              [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # else:
    #     trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],  
    #                              [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    tooth_scan_dir = args.tooth_scan_dir
    tooth_one = os.listdir(tooth_scan_dir)
    mesh_dir_initial = os.path.join(tooth_scan_dir, tooth_one[0])
    mesh3 = trimesh.load(mesh_dir_initial)
    for one in tooth_one:
        if 'stl' in one:
            mesh_dir = os.path.join(tooth_scan_dir, one)
            mesh = trimesh.load(mesh_dir)
            mesh3 += mesh
    mesh3.export(tooth_scan_path)

    voxel_size = 0.6  # means 5cm for the dataset  origin:0.05 origin:0.6 
    source_1, target, source_1_down, target_down, source_1_fpfh, target_fpfh = prepare_dataset(cbct , voxel_size , ismirror)

    source_1_mesh = o3d.io.read_triangle_mesh(tooth_scan_path)
    #global result_ransac
    result_ransac = execute_global_registration(source_1_down, target_down,
                                                source_1_fpfh, target_fpfh,
                                                voxel_size, ismirror)
    print(result_ransac)

    result_icp = refine_registration(source_1_down, target_down,
                                     source_1_fpfh, target_fpfh,
                                     voxel_size, result_ransac)
    print(result_icp)                                          
    result_icp_str = str(result_icp)
    str1 = '='
    str2 = ','

    reg_level = float(result_icp_str[result_icp_str.index(str1)+1:result_icp_str.index(str2)])  
    #import pdb;pdb.set_trace()
    return reg_level, source_1, result_icp, source_1_mesh


if __name__ == '__main__':
    #reg_level, source_1,result_icp.transformation, source_1_mesh = init()
    A = [0]*4 
    A[0], B0, C0, D0 = init('L', 1)
    A[1], B1, C1, D1 = init('U', 1)
    A[2], B2, C2, D2 = init('L', 0)
    A[3], B3, C3, D3 = init('U', 0)
    # import pdb;
    # pdb.set_trace()
    A_sort = sorted(A, reverse=True)
    range = A.index((A_sort[0]))
    # import pdb;pdb.set_trace()
    source_1 = eval(str('B' + str(range)))
    result_icp = eval(str('C' + str(range)))
    source_1_mesh = eval(str('D' + str(range)))
    print(str(range))

    if range == 0 or range == 1:
        print("mirror data\n")
        ismirror = 1
    else:
        print('normal data\n')
        ismirror = 0

    if range == 0 or range == 2:
        print('Lower jaw\n')
        jaw_label = 'L'
    else:
        print('Up jaw\n')
        jaw_label = 'U'
    tooth_scan_dir = args.tooth_scan_dir
    tooth_one = os.listdir(tooth_scan_dir)

    temp_dir = os.path.join(tooth_scan_dir, 'mirror')
    os.makedirs(temp_dir, exist_ok=True)
    for one in tooth_one:
        if 'stl' in one:
            mesh_dir = os.path.join(tooth_scan_dir, one)
            mesh = trimesh.load(mesh_dir)
            if ismirror == 1:
                trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],  
                                         [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
            else:
                trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],  
                                         [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
            mesh = mesh.copy().apply_transform(trans_init)
            mesh = mesh.copy().apply_transform(result_icp.transformation)
            basename = os.path.basename(mesh_dir)
            #basename_name, basename_extend = os.path.splitext(mesh_dir)
            # import pdb;pdb.set_trace()
            mesh.export(os.path.join(temp_dir, basename))

    source_1_temp = copy.deepcopy(source_1)
    result = source_1_temp.transform(result_icp.transformation)
    if ismirror == 1:
        trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], 
                                 [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    else:
        trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], 
                                 [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source_1_mesh.transform(trans_init)
    result_mesh = source_1_mesh.transform(result_icp.transformation)
    result_mesh.compute_vertex_normals()

    result.normals = result_mesh.vertex_normals 
    # import pdb;pdb.set_trace()
    mark = os.path.basename(args.tooth_cbct)
    o3d.io.write_point_cloud('{}/match_normals.ply'.format(args.output_path), result)
    if jaw_label =='U':
        cbct_name = args.tooth_cbct + '/U_all.ply'
    else:
        cbct_name = args.tooth_cbct + '/L_all.ply'
    # import pdb;pdb.set_trace()
    os.system('python Fusion_3.py --tooth_scan_match {}/match_normals.ply  --tooth_cbct_match {} --tooth_cbct_origin {} --ismirror {} --tooth_scan_dir {} --output_path {}'.format(
        args.output_path, cbct_name, args.tooth_cbct_origin, ismirror, temp_dir, args.output_path
    ))
