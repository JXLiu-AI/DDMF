import numpy as np

import onnxruntime

class Segmenter:
    def __init__(self, model_path, enable_cpu_mem_arena=False, enable_mem_pattern=True):
        # With default settings, peak memory usage for enable_cpu_mem_arena and enable_mem_pattern both True is 16GiB
        # peak memory usage for enable_cpu_mem_arena=False and enable_mem_pattern=True is 6.3GiB
        # peak memory usage for enable_cpu_mem_arena=False and enable_mem_pattern=False is 5.5GiB
        
        self.num_categories = 2
        self.c2l_mapping = [0, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28,
                            31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]

        # import the model
        options = onnxruntime.SessionOptions()
        options.enable_cpu_mem_arena = enable_cpu_mem_arena
        options.enable_mem_pattern = enable_mem_pattern
        self.sess = onnxruntime.InferenceSession(model_path, options)
        self.input_name = self.sess.get_inputs()
        self.batch_size = 3
        self.num_pts = 10000
        self.num_parts = 33

    def _get_neighbors(self, inf_indices, knn_indices, k_knn=5):
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=k_knn).fit(inf_indices)
        _, neighbors = knn.kneighbors(knn_indices)
        return neighbors

    def _get_corner(self, mesh):
        # each face has three vertices v1, v2, v3, and a center c
        # get (v1-c), (v2-c), (v3-c)
        return (mesh.triangles - np.expand_dims(mesh.triangles_center, axis=1)).reshape(-1, 9)

    def __del__(self):
        del self.sess

    def _convert_clzz_to_label(self, clzz):
        def convert(x):
            return self.c2l_mapping[x]
        return list(map(convert, clzz))

    def extract_full_segment(self):
        inf_indices = self.inf_indices
        knn_indices = self.knn_indices
        xyz = self.xyz
        seg_res = self.seg_res
        neighbors = inf_indices[self._get_neighbors(xyz[inf_indices], xyz[knn_indices])]
        seg_res[knn_indices] = seg_res[neighbors].mean(axis=1)
        return seg_res

    def segment(self, mesh, category, findices=None):
        mesh.fix_normals()
        if category == 0:
            centroid_trans = np.array([ 1.60083597, -3.74032063,  2.33078412])
        else:
            centroid_trans = np.array([ 0.5687525 , -6.21317224, -6.30563364])

        # translate mesh centroid to mean centroid
        mesh.apply_translation(-centroid_trans)
        xyz = mesh.triangles_center
        corners = self._get_corner(mesh)
        normals = mesh.face_normals

        features = np.hstack((xyz, corners, normals))
        if findices is None:
            findices = np.arange(mesh.faces.shape[0])
            np.random.shuffle(findices)

        num_pts = self.num_pts
        num_parts = self.num_parts

        inf_indices = findices[: (self.batch_size * num_pts)]
        knn_indices = findices[(self.batch_size * num_pts):]
        if len(knn_indices)==0:
            raise ValueError("This mesh has less than", self.batch_size * num_pts + 1, "faces!")

        # construct nn inputs
        nn_feature_input = features[inf_indices].reshape(self.batch_size, num_pts, -1).transpose((0, 2, 1))
        one_hot_categories = np.zeros(self.num_categories)
        one_hot_categories[category] = 1
        nn_categories_input = np.tile(one_hot_categories, [self.batch_size, 1])

        seg_res = np.zeros((mesh.faces.shape[0], num_parts))

        inf_res = self.sess.run([], {self.input_name[0].name: nn_feature_input.astype(np.float32),
                                     self.input_name[1].name: nn_categories_input.astype(np.float32)})[0]

        seg_res[inf_indices] = inf_res.transpose((0, 2, 1)).reshape(-1, num_parts)
        self.xyz = xyz
        self.inf_indices = inf_indices
        self.knn_indices = knn_indices
        self.seg_res = seg_res

        return seg_res[inf_indices]