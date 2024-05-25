from .backbones import Encoder, Sphere2ModelDecoder, Sphere2ModelTargetNetwork,\
      Face2GSColorDecoder, Face2GSShapeDecoder, Face2GSColorTargetNetwork, Face2GSShapeTargetNetwork
from hypercloud.utils.util import get_weights_dir, find_latest_epoch, cuda_setup
from hypercloud.utils.sphere_triangles import generate_sphere_mesh
from hypercloud.datasets import Pts2Nerf
import torch
from os.path import join
from torch.utils.data import DataLoader
from games.mesh_splatting.scene.gaussian_mesh_model import GaussianMeshModel
from utils.sh_utils import RGB2SH
from games.mesh_splatting.utils.graphics_utils import DifferentiableMeshPointCloud
import numpy as np
from utils.graphics_utils import focal2fov, fov2focal
from scene.dataset_readers import (
    getNerfppNorm,
    CameraInfo
)
from PIL import Image
from utils.loss_utils import l1_loss, ssim
from renderer.gaussian_renderer import render
from utils.camera_utils import cameraList_from_camInfos
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.hypercloud_utils import create_embeddings_for_faces
from .backbones import Face2GSDecoder, Face2GSTargetNetwork

ENCODER_WEIGHTS = lambda path, epoch: torch.load(join(path, f'{epoch:05}_E.pth'))
SPHERE2MODEL_DECODER_WEIGHTS = lambda path, epoch: torch.load(join(path, f'{epoch:05}_G.pth'))

MESH_METHOD = lambda config: config['experiments']['sphere_triangles']['method']
MESH_DEPTH = lambda config: config['experiments']['sphere_triangles']['depth']

EPS = 1e-8

USE_SEPARATE_SHAPE_AND_COLOR = False
IS_SHAPE_FREEZED = False

LAMBDA_DSSIM = 0.2 # If constant equals 0 then it is MSE

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_uniform_(m.weight, gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_uniform_(m.weight, gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

def save_training_data(
        gt_image, 
        image, 
        seed,
        vertices,
        faces,
        point_cloud,
        scaling,
):


    gt_image = gt_image.clone().cpu().detach().permute(1,2,0).numpy()
    image = image.clone().cpu().detach().permute(1,2,0).numpy()
    fig, axs = plt.subplots(1,2, figsize=(10,5))

    vertices = vertices.clone().cpu().detach()
    if vertices.shape[0] == 3:
        vertices = vertices.transpose(0,1)
    vertices = vertices.numpy()

    faces = faces.clone().cpu().detach().numpy()


    point_cloud = point_cloud.clone().cpu().detach().transpose(0,1).numpy()


    # RENDERS
    axs[0].set_title("Ground truth")
    axs[0].imshow(gt_image)

    axs[1].set_title("Ours")
    axs[1].imshow(image)

    plt.savefig(f'./debug_data/image_{seed}.png', dpi=300)
    plt.close(fig)

    # MESHES
    fig = plt.figure(figsize=(20,5))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    ax1.set_title('HyperCloud mesh')
    ax2.set_title(f'GS mesh (c={scaling})')
    ax3.set_title('Initial point cloud')

    ax1.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        cmap='viridis',
        alpha=0.4,
        edgecolor='black'
    )


    ax3.scatter3D(point_cloud[:,0], point_cloud[:, 1], point_cloud[:, 2], c='r')

    plt.savefig(f'./debug_data/mesh_{seed}.png', dpi=300)
    plt.close(fig)

def prepare_pcd(raw_alpha, raw_rgb, raw_c, raw_opacity, vertices, faces):
    alpha = torch.relu(raw_alpha) + EPS
    alpha = alpha / alpha.sum(dim=-1, keepdim=True) # normalized
    alpha = alpha.reshape(alpha.shape[0], 1, 3).cuda()

    rgb = torch.relu(raw_rgb) + EPS
    rgb = rgb / rgb.sum(dim=-1, keepdim=True).cuda() # normalized

    c = torch.relu(raw_c) + EPS
    c = c.cuda()

    opacity = torch.sigmoid(raw_opacity).cuda()

    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] = -vertices[:, 1]

    triangles = vertices[faces.clone().detach().long()].float().cuda()

    num_pts = triangles.shape[0]

    points = torch.matmul(
        alpha,
        triangles
    )

    points = points.reshape(num_pts, 3).cuda() # NOTE: Currently only one splat per face is supported

    return DifferentiableMeshPointCloud(
        alpha=alpha,
        points=points,
        colors=rgb,
        normals=np.zeros((num_pts, 3)),
        vertices=vertices,
        faces=faces,
        transform_vertices_function=None,
        triangles=triangles.cuda(),
        opacities=opacity,
        c=c,
    ), c

# NOTE: Possibly in future it will need something to handle white/back background. Now it handles only white.
def gatherCamInfos(cam_poses, images):
    cam_infos = []

    CAMERA_ANGLE_X = 0.6911112070083618

    for idx, image in enumerate(images):
        frame = cam_poses[idx]
        c2w = np.array(frame)
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_pil = Image.fromarray(np.array(image*255.0, dtype=np.byte), "RGB")

        fovy = focal2fov(fov2focal(CAMERA_ANGLE_X, image_pil.size[0]), image_pil.size[1])
        FovY = fovy
        FovX = CAMERA_ANGLE_X

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image_pil,
                            image_path='Unknown', image_name='Unknown', width=image_pil.size[0], height=image_pil.size[1]))
        
    return cam_infos

def get_cameras_extent_radius(cam_poses, images):
    cam_infos = gatherCamInfos(cam_poses, images)
    return cam_infos, getNerfppNorm(cam_infos)['radius']


def hypercloud_training(config, args, pipe):
    args.resolution = 1
    args.data_device = "cuda"
    pipe.debug = False
    pipe.compute_cov3D_python = False
    pipe.convert_SHs_python = False

    device = cuda_setup(config['cuda'], config['gpu'])

    weights_path = get_weights_dir(config)
    epoch = find_latest_epoch(weights_path)

    sphere2model_decoder = Sphere2ModelDecoder(config, device).to(device)
    encoder = Encoder(config).to(device)

    # LOAD PRETRAINED MODELS AND MAKE THEM NON-TRAINABLE
    sphere2model_decoder.load_state_dict(
        SPHERE2MODEL_DECODER_WEIGHTS(
            weights_path,
            epoch
        )
    
    )
    sphere2model_decoder.eval()

    encoder.load_state_dict(
        ENCODER_WEIGHTS(
            weights_path,
            epoch
        )
    )
    encoder.eval()

    # LOAD GS MESH PARAMS DECODER
    if USE_SEPARATE_SHAPE_AND_COLOR:
        face2Colors_decoder = Face2GSColorDecoder(config, device).to(device)
        face2Shape_decoder = Face2GSShapeDecoder(config, device).to(device)
    else:
        face2GS_decoder = Face2GSDecoder(config, device).to(device)

    if USE_SEPARATE_SHAPE_AND_COLOR:
        face2Colors_decoder.apply(weights_init)
        face2Shape_decoder.apply(weights_init)

        face2Colors_decoder.train()
        face2Shape_decoder.train()
    else:
        face2GS_decoder.apply(weights_init)
        face2GS_decoder.train()

    # LOAD DATASET
    dataset_name = config['dataset'].lower()
    assert dataset_name == 'pts2nerf' # Currently only dataset derived from https://github.com/gmum/points2nerf/blob/main/dataset/dataset.py is supported.

    dataset = Pts2Nerf(root_dir=config['data_dir'], classes=config['classes'], debug=True, debug_size=1, images_per_cloud=1)
    batch_size = config['batch_size']
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)

    # GENERATE MESH ON SPHERE
    sphere_vertices, sphere_faces = generate_sphere_mesh(
        MESH_METHOD(config),
        MESH_DEPTH(config)
    )
    sphere_vertices = sphere_vertices.to(device)
    sphere_faces = torch.LongTensor(sphere_faces).to(device)

    gaussian_model = GaussianMeshModel(sh_degree=0)

    if USE_SEPARATE_SHAPE_AND_COLOR:
        optimizer_colors = torch.optim.Adam(face2Colors_decoder.parameters(), lr=0.0003) # Original is 0.001
        optimizer_shape = torch.optim.Adam(face2Shape_decoder.parameters(), lr=0.0005)
        face2Shape_decoder.train()
        face2Colors_decoder.train()
    else:
        optimizer_gs = torch.optim.Adam(face2GS_decoder.parameters(), lr=0.0002)
        face2GS_decoder.train()

    # TO DO: Implement passing epochs from config
    for epoch in range(10000):

        print(50*"#" + f"Epoch {epoch}" + 50*"#")
        loss = None

        # if epoch > 20 and epoch < 300:
        #     IS_SHAPE_FREEZED = True
        #     face2Shape_decoder.eval()
        # else:
        #     IS_SHAPE_FREEZED = False
        
        for i, (point_cloud, images, cam_poses) in enumerate(tqdm(dataloader)):
                x = []
                y = []

                # Move to the device
                point_cloud = point_cloud.to(device).float()

                # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
                if point_cloud.size(-1) == 3:
                    point_cloud.transpose_(point_cloud.dim() - 2, point_cloud.dim() - 1)


                # Compute latent parameters
                codes, mu, logvar = encoder(point_cloud)
                # Compute weights for target network that generates mesh from sphere
                sphere2model_weights = sphere2model_decoder(codes) # NON-TRAINABLE

                # Compute weights for target network that generates gaussian mesh splatting params
                if USE_SEPARATE_SHAPE_AND_COLOR:
                    face2colors_weights = face2Colors_decoder(codes)
                    face2shape_weights = face2Shape_decoder(codes)
                else:
                    face2gs_weights = face2GS_decoder(codes)

                for j in range(batch_size):
                    # Init mesh target network with returned weights
                    sphere2model_target_network = Sphere2ModelTargetNetwork(config, sphere2model_weights[j].to(device)).to(device)

                    # Transform mesh of the sphere
                    transformed_vertices = 2.2 * sphere2model_target_network(sphere_vertices)

                    # For each face we need to compute GS params vector. [N_FACES, 9] -> [N_FACES, N_PARAMS]
                    transformed_faces = transformed_vertices[sphere_faces]
                    transformed_faces = transformed_faces.reshape(transformed_faces.shape[0], -1).to(device)
                    faces_embeddings = create_embeddings_for_faces(transformed_faces, sphere_faces)

                    # Init gaussian params target network
                    if USE_SEPARATE_SHAPE_AND_COLOR:
                        face2colors_target_network = Face2GSColorTargetNetwork(config, face2colors_weights[j].to(device)).to(device)
                        face2shape_target_network = Face2GSShapeTargetNetwork(config, face2shape_weights[j].to(device)).to(device)
                    else:
                        face2gs_target_network = Face2GSTargetNetwork(config, face2gs_weights[j].to(device)).to(device)

                    if USE_SEPARATE_SHAPE_AND_COLOR:
                        gs_colors = face2colors_target_network(faces_embeddings)
                        gs_shapes = face2shape_target_network(faces_embeddings)
                    else:
                        gs_params = face2gs_target_network(faces_embeddings)

                    # We have 7 params in total
                    # 1. First group of 3 are alphas
                    # 2. Second group of 3 is RGB
                    # 3. Next value is scaling parameter (it scales every single gauss separately)
                    # 4. Last value is opacity
                    if USE_SEPARATE_SHAPE_AND_COLOR:
                        raw_rgb, raw_opacity = torch.split(gs_colors, [3, 1], dim=-1)
                        raw_alphas, raw_scaling = torch.split(gs_shapes, [3, 1], dim=-1)
                    else:
                        raw_rgb, raw_opacity, raw_alphas, raw_scaling = torch.split(gs_params, [3,1,3,1], dim=-1)

                    pcd, scaling = prepare_pcd(raw_alphas, raw_rgb, raw_scaling, raw_opacity, transformed_vertices, sphere_faces)
                    cam_infos, radius = get_cameras_extent_radius(cam_poses[j], images[j])
                    
                    # Build gaussian model from pcd parameters returned by `Face2GSParamsTargetNetwork`
                    # We need this to use `GaussianRasterizer`
                    gaussian_model.create_from_pcd(pcd, radius)

                    # PREPARE CAM_INFOS
                    cam_infos, radius = get_cameras_extent_radius(cam_poses[j], images[j])
                    cameraList = cameraList_from_camInfos(cam_infos, resolution_scale=1.0, args=args)
                    random.shuffle(cameraList)
                    
                    # PREPARE RENDERING
                    bg = torch.tensor([1,1,1], dtype=torch.float32, device="cuda")

                    for idx, viewpoint_cam in enumerate(cameraList):
                        render_pkg = render(viewpoint_cam, gaussian_model, pipe, bg)
                        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
                        render_pkg["visibility_filter"], render_pkg["radii"]
                        gt_image = viewpoint_cam.original_image.cuda()

                        try:
                            if idx == 0:
                                save_training_data(
                                    gt_image,
                                    image,
                                    f'(debug)',
                                    transformed_vertices,
                                    sphere_faces,
                                    point_cloud[j],
                                    scaling,
                                )
                        except Exception as e:
                            print(e)

                        x.append(gt_image)
                        y.append(image)
                
                # HERE WE SHOULD DO BACKWARD PASS
                if USE_SEPARATE_SHAPE_AND_COLOR:
                    optimizer_colors.zero_grad()
                    optimizer_shape.zero_grad()
                    face2Shape_decoder.zero_grad()
                    face2Colors_decoder.zero_grad()
                else:
                    optimizer_gs.zero_grad()
                    face2GS_decoder.zero_grad()

                x = torch.stack(x)
                y = torch.stack(y)

                Ll1 = l1_loss(y, x)
                loss = (1.0 - LAMBDA_DSSIM) * Ll1 + LAMBDA_DSSIM * (1.0 - ssim(y, x))

                loss.backward()
                if USE_SEPARATE_SHAPE_AND_COLOR:
                    optimizer_shape.step()
                    optimizer_colors.step()
                else:
                    optimizer_gs.step()
        
        if loss is not None:
            print(f"LOSS {loss.item()}")