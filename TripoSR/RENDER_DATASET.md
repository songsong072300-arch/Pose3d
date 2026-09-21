# DJI Action 4 synthetic dataset rendering

Run commands from `TripoSR/dataset/demo-bin-picking` with the `render` Conda
environment. The default `target` profile is calibrated for the supplied
`frames` sequence: first-person wrists, two wrist-mounted cameras, a light grid
work surface, optional denim context, soft indoor light, and muted video-like
RGB output.

## Preview first

Preview mode writes only RGB images and does not append to `train_pbr`:

```sh
conda run -n render python ../../render_dataset.py 8 \
  --preview-dir ../../preview_target \
  --seed 73
```

The retained sample batch was rendered at `960x720` with 8 Cycles samples.
Change the seed to inspect another pose batch before starting a long run.

## Generate BOP training data

```sh
conda run -n render python ../../render_dataset.py 1000 \
  --seed 1001 \
  --bop-masks
```

This appends RGB, depth, poses, and optional masks to `train_pbr`. Keep
`--scene-profile target` for the provided test sequence. Use a different seed
for each batch.

Useful controls:

- `--garment-probability 0.70`: frequency of denim-like scene context. The
  default is `1.0`, so garments are present in every target-profile image.
- `--front-probability 0.10 --back-probability 0.10`: camera view mix; the
  remaining 80% is split between the two side views.
- `--lower-middle-probability` is retained for command compatibility. The
  current renderer always targets the lower-middle image region while varying
  the horizontal position.
- `--occlusion-probability` is retained for command compatibility. Synthetic
  camera-body finger occluders are currently disabled; all generated
  occlusion labels are `none`.
- `--no-sensor-effects`: disables exposure, color, blur, and JPEG matching.
- `--rear-screen-proxy`: adds a synthetic rear screen only for the old model.
  Leave it disabled for the regenerated Action 4 mesh.
- `--scene-profile generic <cc0textures-path>`: retains the old broad CC0
  background randomization for an ablation or mixed-domain batch.

Do not train only on the existing `000000` and `000001` scenes: they are the
old broad-randomization distribution. Generate a substantial target-profile
batch and either exclude those scenes or keep only a small fraction as a
regularizer.

## Replace the Action 4 model

The regenerated export uses `X/Y/Z = width/height/front-depth`, unlike the old
TripoSR mesh. Convert it with the explicit layout mapping:

```sh
conda run -n render python ../../prepare_bop_model.py \
  --input ../../../3D模型/base.obj \
  --texture ../../../3D模型/texture_diffuse.png \
  --source-layout action4-export \
  --origin center
```

The converter creates timestamped backups, maps the lens to local `-X`, scales
the model to `32.8 x 70.5 x 44.2 mm`, and copies its normal, roughness, and
metallic maps. Preview the result before appending new BOP scenes.

## Visualize corner ground truth

After generating BOP scenes and corner labels, overlay the ordered 3D bounding
box corners on the source RGB images:

```sh
conda run -n render python ../../generate_corner_heatmaps.py .
conda run -n render python ../../visualize_corner_ground_truth.py . \
  --instance-crops
```

Each point label is `instance:corner`. Corner IDs `0-7` use the same stable
binary XYZ order as the heatmap channels and training dataset.
