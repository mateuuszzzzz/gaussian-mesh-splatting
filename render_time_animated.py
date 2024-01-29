#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_animated_renderer import render
import torchvision
import trimesh
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from mesh_splatting.scene.gaussian_mesh_model import GaussianMeshModel
from sklearn.cluster import KMeans

def transform_vertices_function(vertices, c=1):
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] = -vertices[:, 1]
    vertices *= c
    return vertices


def transform_sinus(vertices, t, idxs):
    #vertices[idxs, 0] += 0.0008 * torch.sin(t)  # sinus
    vertices[idxs, 2] += 0.005 * torch.sin(vertices[idxs, 0] * 2 *torch.pi + t)  # sinus
    vertices[idxs, 2] += 0.005 * torch.sin(vertices[idxs, 1] * 5 * torch.pi + t)  # sinus
    #vertices[idxs, 2] += 0.005 * torch.sin(vertices[idxs, 2] * 7 * torch.pi + t)  # sinus
    return vertices


def do_not_transform(vertices, t):
    return vertices


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_time_animated_circle")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    t = torch.linspace(0, 10 * torch.pi, len(views))

    mesh_scene = trimesh.load(f'data/ficus/mesh.obj', force='mesh')
    vertices = mesh_scene.vertices
    vertices = transform_vertices_function(
        torch.tensor(vertices),
    )

    idxs = torch.unique(torch.tensor(mesh_scene.faces).long()[85000:].flatten())
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:
            view_1 = view
        new_vertices = transform_sinus(vertices, t[idx], idxs)
        triangles = new_vertices[torch.tensor(mesh_scene.faces).long()].float().cuda()
        rendering = render(triangles, view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianMeshModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        if hasattr(gaussians, 'update_alpha'):
            gaussians.update_alpha()
        if hasattr(gaussians, 'prepare_scaling_rot'):
            gaussians.prepare_scaling_rot()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        #if not skip_train:
        #     render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)