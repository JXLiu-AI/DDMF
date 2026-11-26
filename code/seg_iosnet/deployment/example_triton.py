from typing import DefaultDict
from prediction_lib_triton import Segmenter
import tempfile
import os
from aligner import Aligner
import trimesh
import numpy as np
from mesh_utils import run_gc_smooth
import argparse
import glob
parser = argparse.ArgumentParser(description = 'CT reconstruction')
parser.add_argument('--model_dir', default='', help='Input onnx model for segmatation')
parser.add_argument('--app_dir', type=str, required=True)
parser.add_argument('--url', type=str, required=True)
parser.add_argument('-in', '--input_dir', default = '', help='Input directory for PNG files')
parser.add_argument('-out', '--output_dir', default = '', help='Output directory for RAW file')
args = parser.parse_args()

os.makedirs(os.path.join(args.output_dir, 'L'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'U'), exist_ok=True)

patient_list = glob.glob(args.input_dir+'/*')

model_path= args.model_dir
# import pdb;pdb.set_trace()
def generate_findices(face_numbers, max_batch, n_points):
        final_list = []
        inf_pts = np.linspace(0, face_numbers - 1, max_batch * n_points).astype(int).tolist()
        for i in range(max_batch):
            final_list += inf_pts[i::max_batch]
        final_list += list(set(range(face_numbers)) - set(inf_pts))

        return np.array(final_list)

# The labels of classification
def convert(x):
    return [0, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28,
            31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48][x]

for mesh_path in patient_list:
    if "L" in mesh_path.split("_"):
        mesh_type = 'l'
    elif "U" in mesh_path.split("_"):
        mesh_type = 'u'
    else:
        print ('incorrect mesh folder name for: ',mesh_path)
        continue

    # h, aligned_mesh_path = tempfile.mkstemp(suffix=".stl")
    # os.close(h)

    # create the output folder
    #out_path = os.path.join(args.output_dir,mesh_path.split("/")[-1].split("_")[0])
    out_path = args.output_dir
    #import pdb;pdb.set_trace()
    print(out_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if mesh_type == 'l':
        aligned_mesh_path = os.path.join(out_path,'l_all.stl')
    if mesh_type == 'u':
        aligned_mesh_path = os.path.join(out_path, 'u_all.stl')
    print(aligned_mesh_path)

    Aligner().align(mesh_path, mesh_type, aligned_mesh_path)
    segmenter = Segmenter(args.url)

    mesh = trimesh.load_mesh(aligned_mesh_path)

    findices = generate_findices(mesh.faces.shape[0], segmenter.batch_size, segmenter.num_pts)

    segmenter.segment(mesh.copy(), (0 if mesh_type=='l' else 1), findices)
    std_inf = segmenter.extract_full_segment()
    smooth_label, std_loss = run_gc_smooth(std_inf, mesh, app_path=args.app_dir)


    smooth_label = smooth_label.astype(int)
    smooth_label = list(map(convert, smooth_label))



    # for each label, get its faces and vertices to form a individal tooth
    for i, l in enumerate(np.unique(smooth_label)):
        dict = DefaultDict(int)
        old_F = mesh.faces[np.where(smooth_label == l)[0]]
        new_F = []
        V = []
        for face in old_F:
            new_idx = []
            for idx in face:
                if not str(mesh.vertices[idx]) in dict:
                    dict[str(mesh.vertices[idx])] = len(dict)
                    V.append(mesh.vertices[idx])
                new_idx.append(dict[str(mesh.vertices[idx])])
            new_F.append(new_idx)
        # create a new mesh of one tooth
        new_mesh = trimesh.Trimesh(vertices=V,faces=new_F)
        # export the tooth to the output path

        #import pdb;pdb.set_trace()
        if 10 < int(str(l)) < 30 :
            tooth_path = os.path.join(out_path+'/U', str(l) + '.stl')
            # import pdb;pdb.set_trace()
            new_mesh.export(tooth_path)
            print(tooth_path)
        elif 30 < int(str(l)) < 50:
            tooth_path = os.path.join(out_path+'/L', str(l) + '.stl')
            new_mesh.export(tooth_path)
            print(tooth_path)
        else:
            tooth_path = os.path.join(out_path, '{}_'.format(mesh_type) + str(l) + '.stl')
            new_mesh.export(tooth_path)
            print(tooth_path)
        #new_mesh.export(tooth_path)
        #print (tooth_path)
