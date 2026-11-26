from lib import *
import os
import trimesh
import logging as log
import scipy
import numpy as np
from tqdm import tqdm
import heapq
from itertools import chain

class seg():
    def __init__(self, args, init=None):
        self.args = args
        self.load_mesh()
        log.info("get mesh")
        self.delete_noise()
        log.info("noise deleted")

        self.old_vertices = self.mesh.vertices
        self.estimate_tooth_n = min(self.old_vertices.shape[0]//10000,32)          
        self.curvature_seg()
        log.info("teeth roughly segmented")

        self.refine_mesh()
        log.info("teeth refined")
        self.centriod = self.mesh.centroid

        self.teeth_mesh_list = self.roughly_merge()
        self.new_vertices = self.mesh.vertices
        log.info("teeth roughly merged")

        self.deleted_points = get_deleted_points(self.old_vertices, sum(self.teeth_mesh_list).vertices)
        log.info("get the deleted points")
        
        self.teeth_pcd_list = self.point_merge()

        self.maxilla, self.mandible = self.up_low_segmentation()


        log.info("point merged")

    def load_mesh(self):
        self.mesh = trimesh.load(self.args.tooth_path)

    def delete_noise(self):
        tooth_split = trimesh.graph.split(self.mesh, only_watertight=False)
        # only get the teeth whose connected points number is larger than the threshold 
        out_tooth =  [k for k in tooth_split if k.vertices.shape[0] > self.args.noise_threshold]
        self.mesh = sum(out_tooth)

    def curvature_seg(self):
        mask = get_removed_mask(self.mesh,self.args.curvature_level,self.args.curvature_seg_threshold)
        self.mesh.update_faces(mask)

    def refine_mesh(self):
        self.mesh = refine_mesh(self.mesh,self.args.refine_threshold,self.args.curvature_level, self.args.removed_threshold)

    # def export_deleted_points(self):
    #     self.deleted_points.export(self.args.deleted_output_path)
    
    def roughly_merge(self):
        components = trimesh.graph.connected_components(edges=self.mesh.face_adjacency,min_len=1000,engine=None)
        mesh_list = self.mesh.submesh(components, only_watertight=False)
        def sort_key(mesh):
            return mesh.faces.shape[0]
        mesh_list = sorted(mesh_list, reverse=True, key=sort_key)
        large_teeth = mesh_list[:self.estimate_tooth_n]
        small_teeth = mesh_list[self.estimate_tooth_n:]

        for small_idx in range(len(small_teeth)):
            tree = scipy.spatial.cKDTree(small_teeth[small_idx].vertices)
            distance = []
            for large_idx in range(self.estimate_tooth_n):
                dist = np.min(np.mean(tree.query(large_teeth[large_idx].vertices, 3)[0],axis=1))
                distance.append(dist)
            merge_target = np.argmin(distance)
            large_teeth[merge_target] += small_teeth[small_idx]


        return large_teeth
    
    def point_merge(self):
        merge_step = self.args.merge_step
        deleted_points = self.deleted_points.vertices
        per_merge_num = deleted_points.shape[0] // merge_step
        last_merge_num = deleted_points.shape[0] - (merge_step - 1) * per_merge_num
        teeth_mesh_list = self.teeth_mesh_list
        # convert mesh to point clouds
        for i in range(len(teeth_mesh_list)):
            #teeth_mesh_list[i].append(trimesh.points.PointCloud(teeth_mesh_list[i].vertices))
            teeth_mesh_list[i] = trimesh.points.PointCloud(teeth_mesh_list[i].vertices)
        # import pdb;pdb.set_trace()
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
                # new_points = trimesh.points.PointCloud(vertices)
                teeth_mesh_list[i] += trimesh.points.PointCloud(vertices)

            deleted_points = np.delete(deleted_points, merge_points_list, axis=0)

        return teeth_mesh_list
    
    def up_low_segmentation(self):
        maxilla = []    #list of idx
        mandible = []
        for tooth_idx in range(len(self.teeth_pcd_list)):
            up_z = np.max(self.teeth_pcd_list[tooth_idx].vertices[:,2])
            low_z = np.min(self.teeth_pcd_list[tooth_idx].vertices[:,2])
            # check whether the tooth belongs to the maxilla or mandible
            if np.abs(up_z-self.centriod[2]) > np.abs(self.centriod[2]-low_z):
                maxilla.append(tooth_idx)
            else:
                mandible.append(tooth_idx)
        return maxilla, mandible
            
    def export_pcd(self, args):
        for i in tqdm(self.maxilla):
            path = os.path.join(args.output_path, 'maxilla')
            os.makedirs(path, exist_ok=True)
            self.teeth_pcd_list[i].export(path + '/%d_U.ply' % i)

        for i in tqdm(self.mandible):
            path = os.path.join(args.output_path, 'mandible')
            os.makedirs(path, exist_ok=True)
            self.teeth_pcd_list[i].export(path + '/%d_L.ply' % i)
        tooth_path_name = os.path.basename(self.args.tooth_path)

        maxilla = sum(self.teeth_pcd_list[i] for i in self.maxilla)
        mandible = sum(self.teeth_pcd_list[i] for i in self.mandible)
        maxilla.export('{}/L_all.ply'.format(args.output_path))
        mandible.export('{}/U_all.ply'.format(args.output_path))
