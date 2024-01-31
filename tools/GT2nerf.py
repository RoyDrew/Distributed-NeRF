import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import transformations as transformations
import json

basepath = "/home/air/multinerf/dataset/2/output.txt"
poses_down = np.loadtxt(os.path.join(basepath, "output.txt")).reshape(-1,4,4)
rgb_path = basepath+"/images/"
rgb_names = [x for x in Path(os.path.join(basepath, "images")).iterdir()]
rgb_names = sorted(rgb_names, key= lambda x:x.stem)

poses = []
c2b = np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]])
for b2w in poses_down:
    c2w = b2w @ transformations.euler_matrix(0,-1.39626,0) @ c2b 
    c2w_RDF = np.linalg.inv(c2b) @ c2w
    c2w_RDF = torch.from_numpy(c2w_RDF).float()
    poses += [c2w_RDF]

RDF_TO_DRB = torch.FloatTensor([[0, 1, 0],
                                [1, 0, 0],
                                [0, 0, -1]])#右下前变为下右后
all_poses = []
for c2w in poses:
    c2w = torch.hstack((
    RDF_TO_DRB @ c2w[:3, :3] @ torch.inverse(RDF_TO_DRB),
    RDF_TO_DRB @ c2w[:3, 3:]))
    c2w = torch.vstack((c2w, torch.Tensor([0.0, 0.0, 0.0, 1.0])))
    all_poses.append(c2w)

json_data = {
  "fl_x": 480.0,
  "fl_y": 480.0,
  "cx": 480.0,
  "cy": 240.0,
  "w": 960.0,
  "h": 480.0, 
  "frames": []
}

for idx, _ in enumerate(tqdm(all_poses)):
    camera_in_drb = all_poses[idx].clone()
    c2w = torch.cat([camera_in_drb[:, 1:2], -camera_in_drb[:, :1], camera_in_drb[:, 2:4]], -1).tolist()
    frame = {
        "file_path": str(rgb_names[idx]),
        "transform_matrix": c2w
    }
    json_data["frames"].append(frame)

with open("/home/air/multinerf/dataset/2/transforms.json", 'w') as f:
    json.dump(json_data, f, indent=4)