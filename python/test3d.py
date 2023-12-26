import copy
import os
import pickle
import sys
sys.path.append('C:/Users/zou/PycharmProjects/building-mapping/')

from facade import Facade

import numpy as np
import open3d as o3d
import time
from gpismap import GPisMap3D
from  util.visualization import show_mesh_3d


def load_file(filepath):
    filefolders = os.listdir(filepath)
    my_dict = {}
    for folder in filefolders:
        f = filepath + folder  # "results/296"
        if os.path.isdir(f):
            label = int(folder)
            files = os.listdir(f)
            my_dict[label] = []
            for file in files:
                fullpath = f + "/" + file
                if not os.path.isdir(fullpath):       # "results/296/building296_faca_0.ply"
                    my_dict[label].append(fullpath)

    return my_dict


def load_map(filepath, id=0):
    # facade groups
    my_files = load_file(filepath)
    building_group = []
    for key in my_files.keys():
        if key < id:
            continue
        dir = filepath + str(key) + '/'
        # file directory
        # facade intermediate results
        facade_dir = dir + 'output/facades_intermediate/'
        if 'output' not in os.listdir(dir):
            continue
        # start parsing
        # load facades from files
        with open(facade_dir + 'groups' + '.pkl', 'rb') as f:  # open file with write-mode
            faca_group = pickle.loads(f.read())
        building_group.append([key, faca_group])

    return building_group


def trans_to_img(pts, resol, f):
    depth = pts[:, 2]
    img_x = pts[:, 0]/depth * f
    img_y = pts[:, 1]/depth * f
    miny = np.min(img_y)
    maxy = np.max(img_y)
    minx = np.min(img_x)
    maxx = np.max(img_x)
    width = maxx - minx
    height = maxy - miny
    col = int(np.round(width/resol) + 1)
    row = int(np.round(height/resol) + 1)
    newpt_rc = np.zeros((len(pts), 2))
    newpt_rc[:, 1] = np.round((img_x - minx) / resol)
    newpt_rc[:, 0] = np.round((img_y - miny) / resol)
    img = np.zeros((row, col))
    newpt_rc = newpt_rc.astype(int)
    img[newpt_rc[:, 0], newpt_rc[:, 1]] = depth

    # surface depth choosing maximum
    value, indices, counts = np.unique(newpt_rc, axis=0, return_counts=True, return_inverse=True)
    id_count = np.where(counts > 1)[0]
    for id in id_count:
        img[value[id, 0], value[id, 1]] = np.max(depth[id == indices])

    cx = -minx / resol
    cy = -miny / resol
    return img, [cx, cy], minx, miny


def generate_test_point(I, f, tr, trans_para, max_resol=0.05, inside_num=2):
    """
        input: mean and var of the surface depth, transformation parameters, max depth resolution
        output: point cloud with world xyz coordinates, normals as mean and var values

                * generate some virtual free points with intervals
    """
    interval = min(max_resol, 0.1)  # at least 0.025 cm cell size
    virtual_num = 3
    pt_all = o3d.geometry.PointCloud()
    r, c = np.where(I > -1)
    I_tmp = copy.deepcopy(I)
    I_tmp[I == 0] = 10
    for j in range(virtual_num):
        surface_dis = j * interval
        # outside
        z = I_tmp[r, c] - surface_dis
        x = (c * trans_para.resol + trans_para.minx) * z / f
        y = (r * trans_para.resol + trans_para.mind) * z / f

        world_xyz = (np.c_[x, y, z] @ np.linalg.inv(trans_para.uvk)) + tr
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(world_xyz)  # ["positions"] = o3d.core.Tensor(world_xyz)
        pt_all = pt_all + pcd
        # inside
        if (j >= 1) and (j <= inside_num):
            z = I_tmp[r, c] + surface_dis
            x = (c * trans_para.resol + trans_para.minx) * z / f
            y = (r * trans_para.resol + trans_para.mind) * z / f
            world_xyz = (np.c_[x, y, z] @ np.linalg.inv(trans_para.uvk)) + tr
            pcd.points = o3d.utility.Vector3dVector(world_xyz)  # ["positions"] = o3d.core.Tensor(world_xyz)
            pt_all = pt_all + pcd
    return np.array(pt_all.points)


def v_to_sdf(v, vvar, lamda):
    v = np.abs(v)
    abn = v <= 1e-7
    v[abn] = 1e-7
    sdf = -1/lamda * np.log(v)  # np.exp(-lamda * d)
    a2 = (1/lamda)**2 / (v ** 2)
    var = np.multiply(a2,  vvar)
    return sdf, var


def main():

    filepath = "C:/Users/zou/source/repos/susiezou/ransac_app/results_0428_amk_new/"
    buildings = load_map(filepath, id=8)
    t_record = {}
    t_record['building'] = []
    t_record['train'] = []
    t_record['test'] = []; t_record['test_number'] = []
    tname = 'GPIS'
    bias = 0.2

    gp = GPisMap3D()
    if gp.loggp:
        tname = 'LogGPIS'
        bias = 0
    # for building in (buildings):
    #     print(f"#frame: {building[0]}")
    #     gp_cloud_dir = filepath + str(building[0]) + '/output/gpismap/meta_data/'
    #     pcd = o3d.io.read_point_cloud(gp_cloud_dir + "pc_depth_" + tname + ".pcd")
    #     n = np.array(pcd.normals)
    #     v = n[:, 0]
    #     vv = n[:, 1]
    #     sdf, var = v_to_sdf(v, vv, 20)
    #     n[:, 0] = sdf
    #     n[:, 1] = var
    #     pcd.normals = o3d.utility.Vector3dVector(n)
    #     o3d.io.write_point_cloud(gp_cloud_dir + "pc_depth_" + tname + ".ply", pcd)

    for building in (buildings):
        print(f"#frame: {building[0]}")
        faca_group = building[1]
        f = 10
        i = 0
        gp_cloud_dir = filepath + str(building[0]) + '/output/gpismap/meta_data/'
        os.makedirs(gp_cloud_dir, exist_ok=True)
        test_xyz = np.array(o3d.io.read_point_cloud("C:/Users/zou/data/localization/julia_map/building_seg/" + str(building[0]) +
                                    "/output/bgk/resolution_0.05/test_pts.pcd").points)
        t_train = {}
        t_train['train'] = []
        t_train['train_number'] = []
        for faca in faca_group:
            pts_global = faca.pts_local @ (faca.trans_para.uvk).T
            trans_local = np.mean(faca.pts_local, axis=0).reshape(1, 3) + [f, 0, 0]
            tr = trans_local @ (faca.trans_para.uvk).T
            uvk = np.c_[faca.trans_para.uvk[:, 1:], faca.trans_para.uvk[:, :1]] @ np.asarray([[1,0,0],[0,-1,0],[0,0,-1]])

            pts_local = (pts_global - tr) @ uvk
            I, cxy, faca.trans_para.minx, faca.trans_para.mind = trans_to_img(pts_local, faca.trans_para.resol, f)
            faca.trans_para.uvk = uvk
            tr = tr.flatten().astype(np.float32)
            Rot = uvk.T.flatten().astype(np.float32)

            fx = fy = f/faca.trans_para.resol
            # can be called only once if the camera param is fixed
            gp.set_cam_param(fx, fy, cxy[0], cxy[1],
                             int(I.shape[1]),
                             int(I.shape[0]))
            tic = time.perf_counter()
            gp.update(I.astype(np.float32), tr, Rot)
            toc = time.perf_counter()
            print(f"Elapsed time: {toc - tic:0.4f} seconds...")
            t_train['train'].append(toc - tic)
            t_train['train_number'].append(len(pts_local))

            # testing:
            # test_xyz = generate_test_point(I, f, tr, faca.trans_para, max_resol=0.05, inside_num=2)
        tic = time.perf_counter()
        sdf, var = show_mesh_3d(gp, (test_xyz[:, 0], test_xyz[:,1], test_xyz[:, 2]), bias=bias)
        toc = time.perf_counter()
        print(f"Test time: {toc - tic:0.4f} seconds...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(test_xyz)  # ["positions"] = o3d.core.Tensor(world_xyz)
        world_n = test_xyz - 0
        world_n[:, 0] = sdf
        world_n[:, 1] = var
        pcd.normals = o3d.utility.Vector3dVector(world_n)
        # save .pcd
        pcd.translate(faca.trans_para.trans)
        o3d.io.write_point_cloud(gp_cloud_dir + "pc_depth_"+tname+".pcd", pcd)
        t_record['building'].append(building[0])
        t_record['train'].append(t_train)
        t_record['test'].append(toc - tic)
        t_record['test_number'].append(len(test_xyz))

        gp.reset()
    input("Press Enter to continue...")

    with open(filepath + 'time_record_' + tname + '.pkl', 'wb') as fp:
        pickle.dump(t_record, fp)

    input("Press Enter to end...")

if __name__ == "__main__":
    main()
