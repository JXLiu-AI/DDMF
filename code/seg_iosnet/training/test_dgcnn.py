import tempfile
import trimesh
import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf
import sys

BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../models'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../part_seg'))

#import dgcnn_seg as model
import part_seg_model_simplified as model

import argparse

BATCH_SIZE=3
NUM_FEATURE=15
NUM_POINT=10000
NUM_CATEGORIES=2

def eval_one_epoch(sess, ops, parent_folder, mode=''):

    for cases in tqdm(os.listdir(parent_folder)):
        patient = cases.split('_')[1]
        #print(patient)
        if os.path.exists(os.path.join(parent_folder, cases, patient+'_u.stl')):
            print(os.path.join(parent_folder, cases, patient+'_u.stl'))
            labels = eval_one_sample(sess, os.path.join(parent_folder, cases, patient + '_u.stl'),ops, 'u', mode)
            np.savetxt(os.path.join(parent_folder, cases, patient + '_u_pred_dgcnn' + mode +'.txt'), labels, fmt='%i')
        if os.path.exists(os.path.join(parent_folder, cases, patient+'_l.stl')):
            print(os.path.join(parent_folder, cases, patient+'_l.stl'))
            labels = eval_one_sample(sess, os.path.join(parent_folder, cases, patient + '_l.stl'), ops, 'l', mode)
            np.savetxt(os.path.join(parent_folder, cases, patient +  '_l_pred_dgcnn' + mode +'.txt'), labels, fmt='%i')

def _get_corner(mesh):
    # each face has three vertices v1, v2, v3, and a center c
    # get (v1-c), (v2-c), (v3-c)
    return (mesh.triangles - np.expand_dims(mesh.triangles_center, axis=1)).reshape(-1, 9)

def eval_one_sample(sess, mesh_path, ops, mesh_type, mode=''):
    # h, aligned_mesh_path = tempfile.mkstemp(suffix=".stl")
    # os.close(h)
    # Aligner().align(mesh_path, mesh_type, aligned_mesh_path)
    # mesh = trimesh.load_mesh(aligned_mesh_path)

    mesh = trimesh.load_mesh(mesh_path)
    mesh.fix_normals()
    # mean_centroids = [[-0.16651961, -2.10824395,  5.60803944], [-0.42354703, -5.87255415, -7.31578397]]
    # category = 0 if mesh_type=='l' else 1
    # mesh.apply_translation(mean_centroids[category] - mesh.centroid)
    if mode is not None:
        mesh.apply_translation(-mesh.centroid)

    xyz = mesh.triangles_center
    normals = mesh.face_normals
    corners = _get_corner(mesh)
    features = np.hstack((xyz, corners, normals))
  #  features = np.hstack((xyz, normals))


    if mesh_type == 'l':
        c = np.array([1.0, 0.0])
    elif mesh_type == 'u':
        c = np.array([0.0, 1.0])

    def generate_findices(face_numbers, max_batch, n_points):
        final_list = []
        inf_pts = np.linspace(0, face_numbers - 1, max_batch * n_points).astype(int).tolist()
        for i in range(max_batch):
            final_list += inf_pts[i::max_batch]
        final_list += list(set(range(face_numbers)) - set(inf_pts))
        return np.array(final_list)

    def _get_neighbors(inf_indices, knn_indices, k_knn=20):
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=k_knn).fit(inf_indices)
        _, neighbors = knn.kneighbors(knn_indices)
        return neighbors

    def _convert_clzz_to_label(labels):
        c2l_mapping = [0, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28,
                       31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]
        def convert(x):
            return c2l_mapping[x]
        return list(map(convert, labels))

    findices = generate_findices(mesh.faces.shape[0], BATCH_SIZE, NUM_POINT)
    inf_indices = findices[:BATCH_SIZE*NUM_POINT]
    knn_indices = findices[BATCH_SIZE*NUM_POINT:]
    features = features[inf_indices].reshape(BATCH_SIZE, NUM_POINT, -1)
    nn_cate = np.tile(c, [BATCH_SIZE, 1])
    print(features.shape)
    feed_dict = {
            ops['points']: features,
            ops['cats']: nn_cate,
            ops['is_training']: False,
        }
    pred_val = sess.run(ops['pred'], feed_dict=feed_dict)
    seg_res = np.zeros((mesh.faces.shape[0], 33))
    seg_res[inf_indices] = pred_val.reshape(-1, 33)
    neighbors = inf_indices[_get_neighbors(xyz[inf_indices], xyz[knn_indices])]
    seg_res[knn_indices] = seg_res[neighbors].mean(axis=1)
    print(seg_res.shape)
    label = np.argmax(seg_res, 1).flatten()

    return _convert_clzz_to_label(label)

def eval(model_path, test_set, mode):
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            points = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, NUM_FEATURE))
            cats = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_CATEGORIES))
            is_training = tf.placeholder(tf.bool, shape=())

            seg_pred = model.get_model(points, cats, is_training,
                                       bn_decay=0.7, cat_num=NUM_CATEGORIES,
                                       part_num=33, batch_size=BATCH_SIZE,
                                       num_point=NUM_POINT, weight_decay=0, k=25, nf=NUM_FEATURE)
            saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False

        with tf.Session(config=config) as sess:
            saver.restore(sess, model_path)

            ops = {'points': points,
                   'cats': cats,
                   'is_training': is_training,
                   'pred': seg_pred
                   }
            eval_one_epoch(sess, ops, test_set, mode)

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default=None, help='path to restore the model [default: None]')
parser.add_argument('--test-path', type=str, default=None, help='path to test set [default: None]')
parser.add_argument('--mode', type=str, default='', help='test with translation or not')
args = parser.parse_args()

eval(args.model_path, args.test_path, args.mode)
