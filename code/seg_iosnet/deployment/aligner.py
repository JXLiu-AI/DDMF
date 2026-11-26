# Adapted from the code by Angel Align.

import glob
import os
import open3d as o3d
import numpy as np
from multiprocessing import Pool
from functools import partial
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, default=None)
    return parser.parse_args()

def tooth_registration_single(vertices_standard, vertices_other):
    
    voxel_size=2.0

    pcd_standard=o3d.geometry.PointCloud()
    pcd_other=o3d.geometry.PointCloud()

    pcd_standard.points=o3d.utility.Vector3dVector(vertices_standard)
    pcd_other.points=o3d.utility.Vector3dVector(vertices_other)

    pcd_standard = pcd_standard.voxel_down_sample(voxel_size=voxel_size)
    pcd_other = pcd_other.voxel_down_sample(voxel_size=voxel_size)

    #color
    pcd_standard.paint_uniform_color([1,0,0])
    pcd_other.paint_uniform_color([0,1,0])

    radius_normal = voxel_size * 2

    pcd_standard.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=15))
    pcd_other.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=15))


    radius_feature = voxel_size * 3
    pcd_standard_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_standard,
                                                     o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature,
                                                                                          max_nn=15))
    pcd_other_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_other,
                                                           o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature,
                                                                                          max_nn=15))
    tmp=0
    for _ in range(5):
        tmp+=1



        distance_threshold = voxel_size * 2.0

        #global registration
        result_g_1 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pcd_other, pcd_standard, pcd_other_fpfh, pcd_standard_fpfh, True, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.6),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 1000), 1)

        result_g_2 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pcd_other, pcd_standard, pcd_other_fpfh, pcd_standard_fpfh, True, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.6),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 1000), 2)
        if result_g_1.transformation[0][0]*result_g_2.transformation[0][0]<0:
            result_g_3 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                pcd_other, pcd_standard, pcd_other_fpfh, pcd_standard_fpfh, True, distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.6),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 1000), 3)
            result_g_4 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                pcd_other, pcd_standard, pcd_other_fpfh, pcd_standard_fpfh, True, distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.6),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 1000), 4)
            result_g_5 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                pcd_other, pcd_standard, pcd_other_fpfh, pcd_standard_fpfh, True, distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.6),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 1000), 5)

            count_1=0
            count_2=0

            if result_g_1.transformation[0][0]*result_g_3.transformation[0][0]>=0:
                count_1+=1
            else:
                count_2+=1

            if result_g_1.transformation[0][0]*result_g_4.transformation[0][0]>=0:
                count_1+=1
            else:
                count_2+=1

            if result_g_1.transformation[0][0]*result_g_5.transformation[0][0]>=0:
                count_1+=1
            else:
                count_2+=1

            if count_1>count_2:
                result_g=result_g_1
            else:
                result_g=result_g_2


        else:
            result_g=result_g_1

        distance_threshold_icp=voxel_size*2.0
        #icp
        result_t=0
        for _ in range(11):
            result = o3d.pipelines.registration.registration_icp(
                pcd_other, pcd_standard, distance_threshold_icp, result_g.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 2000))
            result_t+=result.transformation
        result_t/=11

        threshold_evaluation=2
        evaluation = o3d.pipelines.registration.evaluate_registration(pcd_other, pcd_standard,
                                                            threshold_evaluation,result_t)
        if evaluation.fitness>0.55:
            break
    return result_t, evaluation.fitness


def tooth_registration_multi_standard(vertices_standard_all, path_other, mesh_other_type, path_output):

    mesh_other = o3d.io.read_triangle_mesh(path_other)
    allpoints_other = mesh_other.vertices
    vertices_other = np.unique(allpoints_other,axis=0)
    vertices_standard_5 = vertices_standard_all[mesh_other_type]
    results = []
    for i in range(5):
        result_t, fitness=tooth_registration_single(vertices_standard_5[i], vertices_other)
        results.append((result_t, fitness))
        if fitness > 0.55:
            break
    if results[-1][1] > 0.55:
        transform_t = results[-1][0]
    else:
        transform_t = results[0][0]
        print('Warning:', path_other, 'score is', results[0][1])

    mesh_result=mesh_other.transform(transform_t)
    mesh_result.compute_triangle_normals()
    o3d.io.write_triangle_mesh(path_output, mesh_result)
    return transform_t


def _process_folder(folder, vertices_standard_all):
    # wrapper for multithreading support
    inner = os.listdir(folder)
    for fname in inner:
        if fname.find('stl')==-1:
            continue
        if fname[0] not in ['u','l']:
            print('error processing:', folder)
            continue
        if os.path.exists(os.path.join(folder, fname[0]+'_aligned.stl')):
            print('already processed:', folder)
            continue
        tooth_registration_multi_standard(vertices_standard_all, os.path.join(folder, fname),
                                          fname[0], os.path.join(folder, fname[0]+'_aligned.stl'))
        return

class Aligner:
    def __init__(self):
        # Process all standard
        standard_dir = os.path.join(os.path.dirname(__file__), 'standard')
        L_path = ['standard_L.stl', 'standard_duan_L.stl', 'standard_zhai_L.stl', 'standard_jian_L.stl', 'standard_yuan_L.stl']
        U_path = ['standard_U.stl', 'standard_duan_U.stl', 'standard_zhai_U.stl', 'standard_jian_U.stl', 'standard_yuan_U.stl']
        mesh_standard_L = [o3d.io.read_triangle_mesh(os.path.join(standard_dir, x)) for x in L_path]
        mesh_standard_U = [o3d.io.read_triangle_mesh(os.path.join(standard_dir, x)) for x in U_path]
        vertices_standard_L = [np.unique(x.vertices, axis=0) for x in mesh_standard_L]
        vertices_standard_U = [np.unique(x.vertices, axis=0) for x in mesh_standard_U]
        self.vertices_standard = {'l': vertices_standard_L, 'u': vertices_standard_U}

    def align(self, mesh_path, mesh_type, path_output):
        basename = os.path.basename(mesh_path)
        #np.savetxt("transformation matrix%s.txt"%basename, tooth_registration_multi_standard(self.vertices_standard, mesh_path, mesh_type, path_output))
        # import pdb;
        # pdb.set_trace()
        return tooth_registration_multi_standard(self.vertices_standard, mesh_path, mesh_type, path_output)

    def process_folder(self, folder):
        func = partial(_process_folder, vertices_standard_all=self.vertices_standard)
        dirs = [os.path.join(folder, x) for x in os.listdir(folder)]
        dirs_batch = [dirs[i: i+800] for i in range(0, len(dirs), 800)]
        for idx, data in enumerate(dirs_batch):
            p = Pool()
            p.map(func, data)
            print('Done up to:', (idx+1)*800)
            p.terminate()

if __name__=='__main__':
    args = parse_args()
    a = Aligner()
    a.process_folder(args.folder)
