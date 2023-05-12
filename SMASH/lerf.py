from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import open_clip
import torch
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.ray_samplers import PDFSampler
from nerfstudio.model_components.renderers import DepthRenderer
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.viewer.server.viewer_elements import *
from torch.nn import Parameter
from nerfstudio.utils.colormaps import apply_colormap

from lerf.encoders.image_encoder import BaseImageEncoder
from lerf.lerf_field import LERFField
from lerf.lerf_fieldheadnames import LERFFieldHeadNames
from lerf.lerf_renderers import CLIPRenderer, MeanRenderer

from scipy.spatial.transform import Rotation
import datetime
import os
from PIL import Image

from lerf.xform import transform


# A = np.array([-0.19931519, -0.8966116, -0.53348666])
# B = np.array([-0.3860387, -1.1184605, -0.5293712])
# C = np.array([-0.19470927, -0.8933794, -0.59407806])
# D = np.array([-0.14857963, -0.93668866, -0.5302774])
# ORIG = np.array([-0.34054688, -1.2643669,  -0.44924477])
# REAL_DIST = 0.615

A = np.array([-0.8945317, -0.35725868, -0.3144259])
B = np.array([-1.0468404, -0.33522323, -0.31732386])
C = np.array([-0.89605, -0.3503266, -0.41705683])
D = np.array([-0.9095004, -0.46788028, -0.3203655])
ORIG = np.array([-1.1426075, -0.39756864, -0.30412123])
REAL_DIST = 0.42


@dataclass
class LERFModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: LERFModel)
    clip_loss_weight: float = 0.1
    n_scales: int = 30
    max_scale: float = 1.5
    """maximum scale used to compute relevancy with"""
    num_lerf_samples: int = 24
    hashgrid_layers: Tuple[int] = (12, 12)
    hashgrid_resolutions: Tuple[Tuple[int]] = ((16, 128), (128, 512))
    hashgrid_sizes: Tuple[int] = (19, 19)


class LERFModel(NerfactoModel):
    config: LERFModelConfig

    def populate_modules(self):
        super().populate_modules()

        self.renderer_clip = CLIPRenderer()
        self.renderer_mean = MeanRenderer()

        self.image_encoder: BaseImageEncoder = self.kwargs["image_encoder"]
        self.lerf_field = LERFField(
            self.config.hashgrid_layers,
            self.config.hashgrid_sizes,
            self.config.hashgrid_resolutions,
            clip_n_dims=self.image_encoder.embedding_dim,
        )

        # populate some viewer logic
        # TODO use the values from this code to select the scale
        # def scale_cb(element):
        #     self.config.n_scales = element.value

        # self.n_scale_slider = ViewerSlider("N Scales", 15, 5, 30, 1, cb_hook=scale_cb)

        # def max_cb(element):
        #     self.config.max_scale = element.value

        # self.max_scale_slider = ViewerSlider("Max Scale", 1.5, 0, 5, 0.05, cb_hook=max_cb)

        # def hardcode_scale_cb(element):
        #     self.hardcoded_scale = element.value

        # self.hardcoded_scale_slider = ViewerSlider(
        #     "Hardcoded Scale", 1.0, 0, 5, 0.05, cb_hook=hardcode_scale_cb, disabled=True
        # )

        # def single_scale_cb(element):
        #     self.n_scale_slider.set_disabled(element.value)
        #     self.max_scale_slider.set_disabled(element.value)
        #     self.hardcoded_scale_slider.set_disabled(not element.value)

        # self.single_scale_box = ViewerCheckbox("Single Scale", False, cb_hook=single_scale_cb)

    def get_max_across(self, ray_samples, weights, hashgrid_field, scales_shape, preset_scales=None):
        # TODO smoothen this out
        if preset_scales is not None:
            assert len(preset_scales) == len(self.image_encoder.positives)
            scales_list = torch.tensor(preset_scales)
        else:
            scales_list = torch.linspace(0.0, self.config.max_scale, self.config.n_scales)

        # probably not a good idea bc it's prob going to be a lot of memory
        n_phrases = len(self.image_encoder.positives)
        n_phrases_maxs = [None for _ in range(n_phrases)]
        n_phrases_sims = [None for _ in range(n_phrases)]
        for _, scale in enumerate(scales_list):
            scale = scale.item()
            with torch.no_grad():
                clip_output = self.lerf_field.get_output_from_hashgrid(
                    ray_samples,
                    hashgrid_field,
                    torch.full(scales_shape, scale, device=weights.device, dtype=hashgrid_field.dtype),
                )
            clip_output = self.renderer_clip(embeds=clip_output, weights=weights.detach())

            for i in range(n_phrases):
                probs = self.image_encoder.get_relevancy(clip_output, i)
                pos_prob = probs[..., 0:1]
                if n_phrases_maxs[i] is None or pos_prob.max() > n_phrases_sims[i].max():
                    n_phrases_maxs[i] = scale
                    n_phrases_sims[i] = pos_prob
        return torch.stack(n_phrases_sims), torch.Tensor(n_phrases_maxs)

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        ray_samples_list.append(ray_samples)

        nerfacto_field_outputs, outputs, weights = self._get_outputs_nerfacto(ray_samples)
        lerf_weights, best_ids = torch.topk(weights, self.config.num_lerf_samples, dim=-2, sorted=False)

        def gather_fn(tens):
            return torch.gather(tens, -2, best_ids.expand(*best_ids.shape[:-1], tens.shape[-1]))

        dataclass_fn = lambda dc: dc._apply_fn_to_fields(gather_fn, dataclass_fn)
        lerf_samples = ray_samples._apply_fn_to_fields(gather_fn, dataclass_fn)

        if self.training:
            clip_scales = ray_bundle.metadata["clip_scales"]
            clip_scales = clip_scales[..., None]
            dist = lerf_samples.spacing_to_euclidean_fn(lerf_samples.spacing_starts.squeeze(-1)).unsqueeze(-1)
            clip_scales = clip_scales * ray_bundle.metadata["width"] * (1 / ray_bundle.metadata["fx"]) * dist
        else:
            clip_scales = torch.ones_like(lerf_samples.spacing_starts, device=self.device)

        override_scales = (
            None if "override_scales" not in ray_bundle.metadata else ray_bundle.metadata["override_scales"]
        )
        weights_list.append(weights)
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        lerf_field_outputs = self.lerf_field.get_outputs(lerf_samples, clip_scales)

        if self.training:
            outputs["clip"] = self.renderer_clip(
                embeds=lerf_field_outputs[LERFFieldHeadNames.CLIP], weights=lerf_weights.detach()
            )
            outputs["dino"] = self.renderer_mean(
                embeds=lerf_field_outputs[LERFFieldHeadNames.DINO], weights=lerf_weights.detach()
            )

        if not self.training:
            with torch.no_grad():
                max_across, best_scales = self.get_max_across(
                    lerf_samples,
                    lerf_weights,
                    lerf_field_outputs[LERFFieldHeadNames.HASHGRID],
                    clip_scales.shape,
                    preset_scales=override_scales,
                )
                outputs["raw_relevancy"] = max_across  # N x B x 1
                outputs["best_scales"] = best_scales.to(self.device)  # N

        return outputs

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        LERF overrides this from base_model since we need to compute the max_across relevancy in multiple batches,
        which are not independent since they need to use the same scale
        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        CALIBRATE = False

        if not CALIBRATE:
            # deltas = [0, 90, 180, 270]
            deltas = [0, 60, 120, 180, 240, 300]
            all_pos = []
            all_highest_rel = []
            all_rel_mats = []
            curr = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            dirs = camera_ray_bundle.directions.cpu().numpy()
            w, h, c = dirs.shape

            # hardcode origin
            fixed_orig = np.array([[0.045, -0.045, 0.02]])
            set_orig = np.repeat(fixed_orig, w * h, axis=0)
            set_orig = np.reshape(set_orig, (w, h, c))
            camera_ray_bundle.origins = torch.Tensor(set_orig).to("cuda")

            save = dirs.shape[1] == 300
            # save = False
            if save:
                out_dir = f'outputs/{curr}'
                if not os.path.exists(out_dir): os.makedirs(out_dir)

        if CALIBRATE:
            # TODO(justin) implement max across behavior
            num_rays_per_chunk = self.config.eval_num_rays_per_chunk
            image_height, image_width = camera_ray_bundle.origins.shape[:2]
            num_rays = len(camera_ray_bundle)
            outputs_lists = defaultdict(list)  # dict from name:list of outputs (1 per bundle)
            for i in range(0, num_rays, num_rays_per_chunk):
                start_idx = i
                end_idx = i + num_rays_per_chunk
                ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                outputs = self.forward(ray_bundle=ray_bundle)
                # take the best scale for each query across each ray bundle
                if i == 0:
                    best_scales = outputs["best_scales"]
                    best_relevancies = [m.max() for m in outputs["raw_relevancy"]]
                else:
                    for phrase_i in range(outputs["best_scales"].shape[0]):
                        m = outputs["raw_relevancy"][phrase_i, ...].max()
                        if m > best_relevancies[phrase_i]:
                            best_scales[phrase_i] = outputs["best_scales"][phrase_i]
                            best_relevancies[phrase_i] = m
            # re-render the max_across outputs using the best scales across all batches
            for i in range(0, num_rays, num_rays_per_chunk):
                start_idx = i
                end_idx = i + num_rays_per_chunk
                ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                ray_bundle.metadata["override_scales"] = best_scales
                outputs = self.forward(ray_bundle=ray_bundle)
                # standard nerfstudio concatting
                for output_name, output in outputs.items():  # type: ignore
                    if output_name == "best_scales":
                        continue
                    if output_name == "raw_relevancy":
                        for r_id in range(output.shape[0]):
                            outputs_lists[f"relevancy_{r_id}"].append(output[r_id, ...])
                    else:
                        outputs_lists[output_name].append(output)
            outputs = {}
            for output_name, outputs_list in outputs_lists.items():
                if not torch.is_tensor(outputs_list[0]):
                    # TODO: handle lists of tensors as well
                    continue
                outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore

            # rel = outputs["relevancy_0"].cpu().numpy().squeeze()
            w, h, _ = outputs["rgb"].cpu().numpy().shape
            coord = (int(w/2), int(h/2))
            # print("screen coord: ", coord)
            
            # painting center pixel red
            rgb = outputs["rgb"].cpu().numpy()
            rgb[coord] = [1, 0, 0]
            outputs["rgb"] = torch.Tensor(rgb)

            all_origs = camera_ray_bundle.origins.cpu().numpy()
            all_dirs = camera_ray_bundle.directions.cpu().numpy()
            all_depths = outputs["depth"].cpu().numpy()

            cam_orig = all_origs[coord]
            cam_dir = all_dirs[coord]
            depth = all_depths[coord]

            rel_pos = cam_orig + cam_dir * depth
            print("red dot position: ", rel_pos)

        if not CALIBRATE:
            for delta in deltas:
                # rotate around world up 
                dirs = np.reshape(dirs, (-1, 3))
                axis = [0, 0, 1]
                r = Rotation.from_rotvec(np.radians(delta) * np.array(axis))
                rotated_vectors = r.apply(dirs)
                rotated_vectors = np.reshape(rotated_vectors, (w, h, c))
                camera_ray_bundle.directions = torch.Tensor(rotated_vectors).to("cuda")

                # TODO(justin) implement max across behavior
                num_rays_per_chunk = self.config.eval_num_rays_per_chunk
                image_height, image_width = camera_ray_bundle.origins.shape[:2]
                num_rays = len(camera_ray_bundle)
                outputs_lists = defaultdict(list)  # dict from name:list of outputs (1 per bundle)
                for i in range(0, num_rays, num_rays_per_chunk):
                    start_idx = i
                    end_idx = i + num_rays_per_chunk
                    ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                    
                    print(self.image_encoder.positives)
                    # self.image_encoder.positives = ['bottle']
                    
                    outputs = self.forward(ray_bundle=ray_bundle)
                    # take the best scale for each query across each ray bundle
                    if i == 0:
                        best_scales = outputs["best_scales"]
                        best_relevancies = [m.max() for m in outputs["raw_relevancy"]]
                    else:
                        for phrase_i in range(outputs["best_scales"].shape[0]):
                            m = outputs["raw_relevancy"][phrase_i, ...].max()
                            if m > best_relevancies[phrase_i]:
                                best_scales[phrase_i] = outputs["best_scales"][phrase_i]
                                best_relevancies[phrase_i] = m
                # re-render the max_across outputs using the best scales across all batches
                for i in range(0, num_rays, num_rays_per_chunk):
                    start_idx = i
                    end_idx = i + num_rays_per_chunk
                    ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                    ray_bundle.metadata["override_scales"] = best_scales
                    outputs = self.forward(ray_bundle=ray_bundle)
                    # standard nerfstudio concatting
                    for output_name, output in outputs.items():  # type: ignore
                        if output_name == "best_scales":
                            continue
                        if output_name == "raw_relevancy":
                            for r_id in range(output.shape[0]):
                                outputs_lists[f"relevancy_{r_id}"].append(output[r_id, ...])
                        else:
                            outputs_lists[output_name].append(output)
                outputs = {}
                for output_name, outputs_list in outputs_lists.items():
                    if not torch.is_tensor(outputs_list[0]):
                        # TODO: handle lists of tensors as well
                        continue
                    outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore

                # print(outputs.keys())
                rel = np.zeros_like(outputs["relevancy_0"].cpu().numpy())
                for i in range(len(self.image_encoder.positives)):
                    raw = outputs[f'relevancy_{i}']
                    raw = torch.clip(raw - 0.5, 0, 1)
                    rel += raw.cpu().numpy()

                # rel = outputs["relevancy_0"].cpu().numpy().squeeze()
                w, h, _ = outputs["rgb"].cpu().numpy().shape

                curr_rel = np.max(rel)
                coord = np.unravel_index(rel.squeeze().argmax(), rel.squeeze().shape)
                # print("screen coord: ", coord)
                rgb = outputs["rgb"].cpu().numpy()

                # # painting center pixel red
                # rgb[coord] = [1, 0, 0]
                # outputs["rgb"] = torch.Tensor(rgb)

                if save:    
                    # print(f'max: {np.max(rgb)}, min: {np.min(rgb)}')
                    img = Image.fromarray(np.uint8(rgb * 255))
                    img.save(os.path.join(out_dir, f'{delta}_rgb.png'))
                    rel = torch.Tensor(rel).to("cuda")
                    all_rel_mats.append(rel)
                    # rel = torch.Tensor(rel).to("cuda")
                    # rel_cmap = apply_colormap(rel / (rel.max() + 1e-6), "turbo")
                    # rel_cmap = rel_cmap.cpu().numpy()
                    # img = Image.fromarray(np.uint8(rel_cmap * 255))
                    # img.save(os.path.join(out_dir, f'{delta}_cmap.png'))

                all_origs = camera_ray_bundle.origins.cpu().numpy()
                all_dirs = camera_ray_bundle.directions.cpu().numpy()
                all_depths = outputs["depth"].cpu().numpy()

                cam_orig = all_origs[coord]
                cam_dir = all_dirs[coord]
                depth = all_depths[coord]

                rel_pos = cam_orig + cam_dir * depth
                all_pos.append(rel_pos)
                all_highest_rel.append(curr_rel)

            idx = np.argmax(all_highest_rel)
            pos = all_pos[idx]
            delta = deltas[idx]
            str1 = f"The highest relevancy position out of {len(deltas)} views is angle {delta}" 
            str2 = f"LeRF space coordinate: {pos}"
            print(str1)
            print(str2)

            # colormap normalized across all
            max_across_all_views = np.max(all_highest_rel)
            for i, rel in enumerate(all_rel_mats):
                rel_cmap = apply_colormap(rel / (max_across_all_views + 1e-6), "turbo")
                rel_cmap = rel_cmap.cpu().numpy()
                img = Image.fromarray(np.uint8(rel_cmap * 255))
                img.save(os.path.join(out_dir, f'{deltas[i]}_cmap.png'))

            xformed = transform(pos, A, B, C, D, ORIG, REAL_DIST)
            str3 = f"Spot space coordinate: {xformed}"
            print(str3)

            if save:
                file = open(os.path.join(out_dir, 'query.txt'),'w')
                for word in self.image_encoder.positives:
                    file.write(word+" ")
                file.write("\n")
                file.write(str1)
                file.write("\n")
                file.write(str2)
                file.write("\n")
                file.write(str3)
                file.close()
        return outputs

    def _get_outputs_nerfacto(self, ray_samples: RaySamples):
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        return field_outputs, outputs, weights

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.training:
            unreduced_clip = self.config.clip_loss_weight * torch.nn.functional.huber_loss(
                outputs["clip"], batch["clip"], delta=1.25, reduction="none"
            )
            loss_dict["clip_loss"] = unreduced_clip.sum(dim=-1).nanmean()
            unreduced_dino = torch.nn.functional.mse_loss(outputs["dino"], batch["dino"], reduction="none")
            loss_dict["dino_loss"] = unreduced_dino.sum(dim=-1).nanmean()
        return loss_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["lerf"] = list(self.lerf_field.parameters())
        return param_groups
