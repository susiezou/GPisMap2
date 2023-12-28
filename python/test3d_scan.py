import copy
import json
import os
import pickle
import sys
sys.path.append('C:/Users/zou/PycharmProjects/building-mapping/')

import numpy as np
import open3d as o3d
import time
from gpismap import GPisMap3D
from  util.visualization import show_mesh_3d


def trans_to_img(pts, resolx=0.00175,resoly=0.00175):
    miny = np.min(pts[:, 1])
    minx = np.min(pts[:, 0])
    cc = np.round((pts[:, 0] - minx) / resolx).astype(int)
    rr = np.round((pts[:, 1] - miny) / resoly).astype(int)
    img = np.zeros((rr.max() + 1, cc.max() + 1))
    img[rr, cc] = pts[:, 2]

    cx = -minx / resolx
    cy = -miny / resoly
    return img, [cx, cy], minx, miny


def v_to_sdf(v, vvar, lamda):
    v = np.abs(v)
    abn = v <= 1e-7
    v[abn] = 1e-7
    sdf = -1/lamda * np.log(v)  # np.exp(-lamda * v)
    a2 = (1/lamda)**2 / (v ** 2)
    var = np.multiply(a2,  vvar)
    return sdf, var


def read_ply_file(
    ply_file,
    dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("attribute", "<f4")]),
):
    end_flag = "end_header"
    num_flag = "element vertex "
    normal_flag = "comment normal "
    nx, ny, nz = "0", "0", "0"
    if ".ply" in (ply_file):
        with open(ply_file, "rb") as f:
            strLine = f.readline().decode("utf-8")
            while strLine != "":
                strLine = f.readline().decode("utf-8")

                if end_flag in strLine:
                    break

                if num_flag in strLine:
                    numPts = int(strLine.split(" ")[-1])

                if normal_flag in strLine:
                    nx = strLine.split(" ")[-3]
                    ny = strLine.split(" ")[-2]
                    nz = strLine.split(" ")[-1]

            data = np.fromfile(f, dtype=dtype, count=numPts)

    else:
        raise Exception("Can't find ply file!")
    return data, np.asarray([float(nx), float(ny), float(nz)])

def train_gp(gp, t_record):
    # training GP:
    tic = time.perf_counter()
    gp.trainGPs()
    toc = time.perf_counter()
    print(f"Train time: {toc - tic:0.4f} seconds...")
    t_record['frame'].append('-10')
    t_record['train'].append(toc - tic)


def loop_single_frame(gp, data, traj, center, t_record):
    cc = 0
    aer = np.c_[data['x'],data['y'], data['z']]
    frameid = traj.keys()
    for k in frameid:
        if (int(k) <930) or (int(k)>1000):
            continue
        rx = 0.00175 * 3
        ry = 0.0175 * 1.2
        print(f"#frame: {k}")
        t_record['frame'].append(k)
        tr = np.array(traj[k]['ego_transformation'])[:3]
        valid = data['frameID'] == int(k)
        dataz = aer[valid]
        if len(dataz) <= 0:
            continue
        vec = center - tr[:3]
        # rotate azimuth
        a0 = np.arctan2(vec[1], vec[0])
        dataz[:, 0] -= a0
        valid2 = np.abs(dataz[:, 0]) < 0.25 * 3.14
        datav = dataz[valid2]

        r_sub = np.cos(datav[:, 1]) * datav[:, 2]
        zloc = np.cos(datav[:, 0]) * r_sub
        u = np.tan(datav[:, 0])
        v = np.tan(datav[:, 1])/np.cos(datav[:, 0])

        img, cxy, minx, miny = trans_to_img(np.c_[u, v, zloc], resolx=rx,resoly=ry)

        tr = tr.flatten().astype(np.float32)
        Rot = (np.array([[0,1,0],[0,0,1],[1,0,0]]) @
               np.array([[np.cos(-a0), -np.sin(-a0),0],[np.sin(-a0), np.cos(-a0),0], [0, 0, 1]])).flatten().astype(np.float32)
        gp.set_cam_param(rx, ry, cxy[0], cxy[1], img.shape[1], img.shape[0])
        tic = time.perf_counter()
        # gp.update_scan(img.astype(np.float32), tr, Rot)
        gp.add_scan(img.astype(np.float32), tr, Rot)
        toc = time.perf_counter()
        print(f"Elapsed time: {toc - tic:0.4f} seconds...")
        t_record['train'].append(toc - tic)
        t_record['train_number'].append(len(dataz[valid2]))
        cc += 1
        if cc == 50:
            cc = 0
            train_gp(gp, t_record)  # training GP
    return gp, k, cc

def loop_batch_frame(gp, data, traj, center, xy, t_record, bsize=20):
    cc = 0;
    ids = np.array(data.normals)[:, 2]
    xyz = np.array(data.points)
    frameid = list(traj.keys())
    for k in frameid[::bsize]:
        rx = 0.00175 * 0.5
        ry = 0.0175 * 0.2
        print(f"#frame: {k}")
        t_record['frame'].append(k)

        ik = int(k)
        tr = np.array([0., 0., 0.])
        valid = np.zeros(len(ids), dtype=bool)
        for i in range(bsize):
            iki = ik + i
            s = f"{iki}"
            tr += np.array(traj[s]['ego_transformation'])[:3]
            valid = valid | (ids == iki)

        tr /= bsize
        dataz = xyz[valid]
        if len(dataz) <= 0:
            continue
        vec = center - tr[:3]
        if np.linalg.norm(vec) > 100:
            continue
        a0 = np.arctan2(vec[1], vec[0])

        tr = tr.flatten().astype(np.float32)
        Rot = (np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]) @
               np.array([[np.cos(-a0), -np.sin(-a0), 0], [np.sin(-a0), np.cos(-a0), 0], [0, 0, 1]])).flatten().astype(
            np.float32)

        uvk = Rot.reshape(3, 3).T
        pts_local = (dataz - tr) @ uvk
        xy_local = (xy - tr) @ uvk
        pts = pts_local[pts_local[:, 2] > 0]

        u = pts[:, 0] / pts[:, 2]
        v = pts[:, 1] / pts[:, 2]
        ulimit = xy_local[:,0] / xy_local[:,2]
        valid2 = (u < np.max(ulimit) + 0.04) & (u > np.min(ulimit) - 0.04)

        img, cxy, minx, miny = trans_to_img(np.c_[u, v, pts[:, 2]][valid2], resolx=rx, resoly=ry)

        gp.set_cam_param(rx, ry, cxy[0], cxy[1], img.shape[1], img.shape[0])
        tic = time.perf_counter()
        # gp.update_scan(img.astype(np.float32), tr, Rot)
        gp.add_scan(img.astype(np.float32), tr, Rot)
        toc = time.perf_counter()
        print(f"Elapsed time: {toc - tic:0.4f} seconds...")
        t_record['train'].append(toc - tic)
        t_record['train_number'].append(len(pts[valid2]))
        cc += 1
        if cc == 10:
            cc = 0
            train_gp(gp, t_record)  # training GP
    return gp, k, cc

def main():
    filepath = 'D:/carla/dense_no_occlusion/'
    with open('D:/carla/recording_Town10HD_Opt_2023-09-05_14_38.json', 'r') as file:
        traj = json.load(file)

    # data, _ = read_ply_file(filepath + 'scanstrips/all_frames.ply',
    #                dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("labels", "I"), ("frameID", "I")]))
    data = o3d.io.read_point_cloud(filepath + 'train_pts/pc_no_normal_noisy.ply')
    obj = o3d.io.read_point_cloud(filepath + 'train_pts/building_pc_noisy_sample2.ply')
    
    t_record = {}
    t_record['frame'] = []
    t_record['train'] = []; t_record['train_number'] = []
    t_record['test'] = []; t_record['test_number'] = []
    tname = 'GPIS'; bias = 0.2

    # set mapping box
    center = obj.get_center()
    minb = obj.get_min_bound()
    maxb = obj.get_max_bound()
    xx, yy = np.meshgrid(np.c_[minb[0], maxb[0]], np.c_[minb[1], maxb[1]])
    xy = np.c_[xx.flatten(), yy.flatten()]

    test_xyz = np.array(o3d.io.read_point_cloud(filepath + 'test_pts/resolution_0.05/test_pts_sample2_sdf.ply').points)
    gp_cloud_dir = filepath + '/output/gpismap/meta_data/frames/'
    os.makedirs(gp_cloud_dir, exist_ok=True)

    gp = GPisMap3D()
    if gp.loggp:
        tname = 'LogGPIS'
        bias = 0

    # loop for each frame
    gp, k, cc = loop_batch_frame(gp, data, traj, center, xy, t_record)

    if cc != 0:
        train_gp(gp, t_record)

    # testing:
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
    o3d.io.write_point_cloud(gp_cloud_dir + "pc_"+ k +"_"+tname+".pcd", pcd)
    t_record['test'].append(toc - tic)
    t_record['test_number'].append(len(test_xyz))

    gp.reset()
    input("Press Enter to continue...")

    with open(gp_cloud_dir + 'time_record_' + tname + '.pkl', 'wb') as fp:
        pickle.dump(t_record, fp)

    input("Press Enter to end...")

if __name__ == "__main__":
    main()
