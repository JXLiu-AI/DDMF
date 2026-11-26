import numpy as np
import tempfile
import shutil
from os.path import join
import subprocess
def create_colors():
    # 20 high contrast colors
    colors = [[230, 25, 75,255],[60, 180, 75,255],[255, 225, 25,255],\
            [0, 130, 200,255],[245, 130, 48,255],[145, 30, 180,255],[70, 240, 240,255],\
            [240, 50, 230,255],[210, 245, 60,255],[250, 190, 190,255],[0, 128, 128,255],\
            [230, 190, 255,255],[170, 110, 40,255],[255, 250, 200,255],[128, 0, 0,255],\
            [170, 255, 195,255],[128, 128, 0, 255]]
    #np.random.shuffle(colors)
    # gum color
    colors = [[255,255,255,255]] + colors
    return colors

def colored_mesh(mesh, label):
    COLORS = create_colors()
    mcopy = mesh.copy()
    for i, l in enumerate(np.unique(label)):
        mcopy.visual.face_colors[np.where(label == l)[0]] = COLORS[i % 18]
    return mcopy

def run_gc_smooth(preds, tmesh, nl=33, lmd=130, app_path=None):
    # export off to temp
    BASE_DIR = tempfile.mkdtemp()
    tmesh.export(join(BASE_DIR, 'mesh.off'))
    np.savetxt(join(BASE_DIR, 'pred_probs'), preds, delimiter=',')
    np.savetxt(join(BASE_DIR, 'neighbors'), tmesh.face_adjacency, fmt='%i', delimiter=',')

    cmd = [app_path, str(lmd), str(nl), \
           join(BASE_DIR, 'mesh.off'), join(BASE_DIR, "pred_probs"), join(BASE_DIR, "neighbors"), \
           join(BASE_DIR, 'output_label'), str(tmesh.face_adjacency.shape[0])]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    output, err = p.communicate()
    p.wait()
    if err:
        print("err:", err)
        shutil.rmtree(BASE_DIR)
        return None
    output = output.decode("utf-8").split("\n")
    res = np.loadtxt(join(BASE_DIR, 'output_label'), delimiter=',')[:,1]
    shutil.rmtree(BASE_DIR)

    # return result and final loss
    return res, output[-4].split()[-1]

def mesh_inf_compare(std, inf):
    np.seterr(divide='ignore', invalid='ignore') # allow for div by zero to generate nan
    total = np.array([0] * 33)
    same = np.array([0] * 33)
    totali = np.array([0] * 33)
    samei = np.array([0] * 33)
    for idx, s in enumerate(std):
        i = inf[idx]
        if i==s:
            same[s]+=1
        total[s] += 1
    # reverse comparison
    for idx, s in enumerate(inf):
        i = std[idx]
        if i==s:
            samei[s]+=1
        totali[s] += 1
    res = same / total
    resi = samei / totali
    res = np.minimum(res, resi)
    return [1.0 if np.isnan(p) else p for p in res]

def get_gum_tooth_boundary(mesh, label):
    '''
    Get gum and tooth boundary indices.

    Args:
        label: 1-d int array of length len(mesh.faces) containing labels
        mesh: Trimesh object
    Returns:
        boundary_ids: 1-d numpy array countaining boundaries' face_id
    '''

    adj_label_pair = label[mesh.face_adjacency]
    boundary_mask = np.logical_xor(adj_label_pair[:, 0], adj_label_pair[:, 1])
    boundary_ids = mesh.face_adjacency[boundary_mask].ravel()
    boundary_ids = np.unique(boundary_ids)
    return boundary_ids
