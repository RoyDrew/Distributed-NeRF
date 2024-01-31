import numpy as np
import jax.numpy as jnp
from easydict import EasyDict as edict
from jax import vmap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import transformations as transformations

def procrustes_analysis(X0,X1): # [N,3]
    # translation
    t0 = jnp.mean(X0, axis=0, keepdims=True)
    t1 = jnp.mean(X1, axis=0, keepdims=True)
    X0c = X0-t0
    X1c = X1-t1
    # scale
    s0 = jnp.sqrt(jnp.mean(jnp.sum(X0c**2, axis=-1)))
    s1 = jnp.sqrt(jnp.mean(jnp.sum(X1c**2, axis=-1)))
    X0cs = X0c/s0
    X1cs = X1c/s1
    # rotation (use double for SVD, float loses precision)
    U, S, V = jnp.linalg.svd(jnp.array(X0cs.T @ X1cs, dtype=jnp.float64))
    R = jnp.array(U @ V.T, dtype=jnp.float32)
    if jnp.linalg.det(R)<0:
        R.at[2].mul(-1)
    sim3 = edict(t0=t0[0],t1=t1[0],s0=s0,s1=s1,R=R)
    return sim3

def to_inv(pose):
    bottom = jnp.array([[0,0,0,1]])
    matrix = jnp.concatenate([pose, bottom], axis=0)
    matrix = jnp.linalg.inv(matrix)
    return matrix

def read_gt_v0():
    txt_file = "/home/air/multinerf/dataset/NewYork_2_26items_of_tails/output.txt"
    GT_data = np.loadtxt(txt_file)
    GT_data = GT_data.reshape(-1, 4, 4)
    poses = []
    c2b = np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]])
    for b2w in GT_data:
        c2w = b2w @ transformations.euler_matrix(0,-1.39626,-1.39626) @ c2b 
        #c2w = b2w @ transformations.euler_matrix(0,-1.39626,0) @ c2b 
        #c2w = b2w @ transformations.euler_matrix(0,0,0) @ c2b 
        c2w_RDF = np.linalg.inv(c2b) @ c2w
        poses += [c2w_RDF]

    RDF_TO_DRB = np.array([[0., 1., 0.],
                                [1, 0, 0],
                                [0, 0, -1]])#右下前变为下右后
    #NeRF (right, up, back) frame.
    RDF_TO_DRB = np.array([[1, 0, 0.],[0, -1, 0],[0, 0, -1]])
    all_poses = []
    for c2w in poses:
        c2w = c2w.copy()
        c2w = np.concatenate((
        RDF_TO_DRB @ c2w[:3, :3] @ np.linalg.inv(RDF_TO_DRB),
        RDF_TO_DRB @ c2w[:3, 3:]), 1)
        c2w = np.concatenate((c2w, np.array([0.0, 0.0, 0.0, 1.0]).reshape(1,4)), 0)
        all_poses.append(c2w)
    
    GT_pose = np.array(all_poses).copy()
    scale = 1 / np.max(np.abs(GT_pose[:, :3, 3].reshape(-1)))
    first_GT = GT_pose[0, :3, 3]
    
    GT_pose[:, :3, 3] -= first_GT
    GT_pose[:, :3, 3] *= scale
    return GT_pose

def read_gt():
    txt_file = "/home/air/multinerf/dataset/NewYork_2_26items_of_tails/output.txt"
    GT_data = np.loadtxt(txt_file)
    GT_data = GT_data.reshape(-1, 4, 4)

    poses = []
    for pose in GT_data:
        c2w = pose @ transformations.euler_matrix(0.1, -0.8, -0.1)
        poses += [c2w]

    GT_pose = np.array(poses) @ np.diag([1, -1, -1, 1])
    
    scale = 1 / np.max(np.abs(GT_pose[:, :3, 3].reshape(-1)))
    first_GT = GT_pose[0, :3, 3]
    
    GT_pose[:, :3, 3] -= first_GT
    GT_pose[:, :3, 3] *= scale
    return GT_pose

def plot_cameras(origin_cameras, refined_cameras):
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    
    min_x, min_y, min_z = np.inf, np.inf, np.inf
    max_x, max_y, max_z = -np.inf, -np.inf, -np.inf
    
    GT_pose = read_gt()
    
    center_gt = GT_pose[:, :3, 3]
    
    
    center_origin = origin_cameras[:, :3, 3]
    sim3 = procrustes_analysis(center_gt, center_origin)

    center_aligned = (center_origin-sim3.t1)/sim3.s1@sim3.R.T*sim3.s0+sim3.t0
    R_aligned = origin_cameras[..., :3, :3] @ sim3.R.T
    t_aligned = (-R_aligned@center_aligned[..., None])[..., 0]
    origin_aligned = jnp.concatenate([R_aligned, t_aligned[:, :, None]], axis=-1)
    
    center_refined = refined_cameras[:, :3, 3]
    sim3 = procrustes_analysis(center_gt, center_refined)
    
    center_aligned = ((center_refined-sim3.t1)/sim3.s1)@sim3.R.T*sim3.s0+sim3.t0
    R_aligned = refined_cameras[..., :3, :3] @ sim3.R.T
    t_aligned = (-R_aligned@center_aligned[..., None])[..., 0]
    refined_aligned = jnp.concatenate([R_aligned, t_aligned[:, :, None]], axis=-1)
    
    
    """
    def update_limits(vertex):
        nonlocal min_x, min_y, min_z, max_x, max_y, max_z
        min_x = min(min_x, vertex[0])
        min_y = min(min_y, vertex[1])
        min_z = min(min_z, vertex[2])
        max_x = max(max_x, vertex[0])
        max_y = max(max_y, vertex[1])
        max_z = max(max_z, vertex[2])

    
    def extrinsic2pyramid(extrinsic, ax, color='r', focal_len_scaled=0.05, aspect_ratio=0.3):
        vertex_std = np.array([[0, 0, 0, 1],
                               [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, -focal_len_scaled* aspect_ratio, focal_len_scaled, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        faces = [[vertex_transformed[j, :-1] for j in face] for face in [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 2, 3, 4]]]
        
        for vertex in vertex_transformed:
            update_limits(vertex[:-1])
        
        # Create the Poly3DCollection
        poly3d = Poly3DCollection(faces, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35)
        ax.add_collection3d(poly3d)

    for matrix in origin_aligned[:4]:
        bottom = jnp.array([[0,0,0,1]])
        matrix = jnp.concatenate([matrix, bottom], axis=0)
        matrix = jnp.linalg.inv(matrix)
        extrinsic2pyramid(matrix, ax, 'red')
        
    for matrix in refined_aligned[:4]:
        bottom = jnp.array([[0,0,0,1]])
        matrix = jnp.concatenate([matrix, bottom], axis=0)
        matrix = jnp.linalg.inv(matrix)
        
        extrinsic2pyramid(matrix, ax, 'green')
    for matrix in GT_pose[:4]:
        matrix = jnp.array(matrix)
        #matrix = jnp.linalg.inv(matrix)
        extrinsic2pyramid(matrix, ax, 'blue')
    """
    gt_positions = GT_pose[:, :3, 3]
    origin_aligned = vmap(to_inv, in_axes=0)(origin_aligned)
    colmap_positions = origin_aligned[:, :3, 3]
    
    refined_aligned = vmap(to_inv, in_axes=0)(refined_aligned)
    refined_positions = refined_aligned[:, :3, 3]
    
    ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], label='GT', color='blue')
    ax.plot(colmap_positions[:, 0], colmap_positions[:, 1], colmap_positions[:, 2], label='refined', color='green')
    ax.plot(refined_positions[:, 0], refined_positions[:, 1], refined_positions[:, 2], label='colmap', color='red')
    
    # 方法1
    ax.set_xlim(-1, 0)
    ax.set_ylim(-0.03, 0.03)
    ax.set_zlim(-0.03, 0.03)
    
    #ax.set_xlim(min_x, max_x)
    #ax.set_ylim(min_y, max_y)
    #ax.set_zlim(min_z, max_z)

    #方法2
    #ax.set_box_aspect([max_x-min_x, max_y - min_y, max_z-min_z])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Positions and Orientations')
    ax.legend()
    plt.show()

def plot_save_poses(path, poses, refined_poses=None, ep=None):
    # get the camera meshes
    _, cam = vmap(get_camera_cone, in_axes=0)(poses)
    if refined_poses is not None:
        _, cam_ref = vmap(get_camera_cone, in_axes=0)(refined_poses)
    else:
      cam_ref = cam
    cam = np.array(cam)
    cam_ref = np.array(cam_ref)
    # set up plot window(s)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("epoch {}".format(ep), pad=0)
    min_x, max_x = cam_ref[:, :, 0].min(), cam_ref[:, :, 0].max() # xsy
    min_y, max_y = cam_ref[:, :, 1].min(), cam_ref[:, :, 1].max() # xsy
    min_z, max_z = cam_ref[:, :, 2].min(), cam_ref[:, :, 2].max() # xsy
    setup_3D_plot(ax, elev=45, azim=35, lim=edict(x=(2*np.sign(min_x)*np.abs(min_x),np.sign(max_x)*np.abs(max_x)), y=(2*np.sign(min_y)*np.abs(min_y),np.sign(max_y)*np.abs(max_y)), z=(2*np.sign(min_z)*np.abs(min_z),np.sign(max_z)*np.abs(max_z))))  # (x=(-3,3), y=(-3,3), z=(-3,2.4))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=0.95, wspace=0, hspace=0)
    plt.margins(tight=True, x=0, y=0)
    # plot the cameras
    N = len(cam)
    ref_color = (0.7,0.2,0.7)
    pred_color = (0,0.6,0.7)
    ax.add_collection3d(Poly3DCollection([v for v in cam_ref], alpha=0.2, facecolor=ref_color))
    for i in range(N):
        ax.plot(cam_ref[i,:,0], cam_ref[i,:,1], cam_ref[i,:,2], color=ref_color, linewidth=0.5)
        ax.scatter(cam_ref[i,4,0], cam_ref[i,4,1], cam_ref[i,4,2], color=ref_color,s=20)
    if ep==0:
        png_fname = "{}/GT.png".format(path)
        plt.savefig(png_fname,dpi=75)
    ax.add_collection3d(Poly3DCollection([v for v in cam], alpha=0.2, facecolor=pred_color))
    for i in range(N):
        ax.plot(cam[i,:,0], cam[i,:,1], cam[i,:,2], color=pred_color, linewidth=1)
        ax.scatter(cam[i,4,0], cam[i,4,1], cam[i,4,2], color=pred_color, s=20)
    # ax.autoscale_view()     # xsy
    png_fname = "{}/{}.png".format(path, ep)
    plt.show()
    # clean up
    #plt.clf()
    #return png_fname

def get_camera_cone(pose):
    # pose 3x4
    depth=0.00001
    vertices = jnp.array([[-0.5,-0.5,1],
                             [0.5,-0.5,1],
                             [0.5,0.5,1],
                             [-0.5,0.5,1],
                             [0,0,0]])*depth
    vertices = jnp.concatenate([vertices, jnp.ones_like(vertices[:, :1])], axis=-1)
    
    # vertices forms a camera facing z direction at origin.
    bottom = jnp.array([[0,0,0,1]])
    _pose = jnp.linalg.inv(jnp.concatenate([pose, bottom], axis=0))
    vertices = jnp.matmul(vertices, _pose)
    # converts vertices to world coordinate system.
    wireframe = vertices[:, :3]
    return vertices, wireframe

def setup_3D_plot(ax,elev,azim,lim=None):
    ax.xaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.yaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.zaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.xaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.yaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.zaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.zaxis.set_tick_params(labelsize=8)
    ax.set_xlabel("X",fontsize=16)
    ax.set_ylabel("Y",fontsize=16)
    ax.set_zlabel("Z",fontsize=16)
    ax.set_xlim(lim.x[0],lim.x[1])
    ax.set_ylim(lim.y[0],lim.y[1])
    ax.set_zlim(lim.z[0],lim.z[1])
    ax.view_init(elev=elev,azim=azim)