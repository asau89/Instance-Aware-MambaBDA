# BRIGHT Quick Guide (ChangeMamba BDA)

This note is tailored to the BRIGHT/xBD-style building damage assessment workflow using the slim scripts in this folder.

## Files in this folder
- **train_MambaBDA_bright.py**: Baseline training with cross-entropy + Lovasz losses for localization and damage classification.
- **train_MambaBDA_bright_ICR.py**: Baseline plus instance consistency regularization (ICR) on feature maps after a warmup.
- **train_MambaBDA_bright_ForceDirected.py**: Baseline plus force-directed instance consistency loss on high-res features (compact + separation).
- **infer_MambaBDA_BRIGHT.py**: Single-GPU inference and evaluation; writes localization masks and color-mapped damage predictions per sample.

## Dataset layout (BRIGHT/xBD style)
```
${BRIGHT_ROOT}
├── train
│   ├── images
│   │   ├── sample_00000000_pre_disaster.png
│   │   ├── sample_00000000_post_disaster.png
│   │   └── ...
│   └── targets
│       ├── sample_00000000_pre_disaster_target.png
│       ├── sample_00000000_post_disaster_target.png
│       └── ...
├── val
│   ├── images
│   └── targets
├── test
│   ├── images
│   └── targets
├── train.txt   # one stem per line, e.g., sample_00000000
├── val.txt
└── test.txt
```
Notes:
- Stems in the txt files must match the shared prefix of the paired pre/post image and target names.
- Targets carry both localization and damage labels as expected by `MultimodalDamageAssessmentDatset`.
- If your data uses `.tif`, pass `suffix='.tif'` where applicable (see the test loader in the scripts).

## Minimal training commands
Set your paths, then pick one script. Common flags: `--cfg`, `--pretrained_weight_path`, dataset paths, list paths, `--train_batch_size`, `--crop_size`, `--learning_rate`, `--weight_decay`, `--resume` (optional), `--seed`.

Baseline:
```bash
python train_MambaBDA_bright.py \
  --cfg /path/to/vssm_base_224.yaml \
  --pretrained_weight_path /path/to/pretrained.pth \
  --dataset BRIGHT \
  --train_dataset_path /data/bright/train \
  --val_dataset_path /data/bright/val \
  --test_dataset_path /data/bright/test \
  --train_data_list_path /data/bright/train.txt \
  --val_data_list_path /data/bright/val.txt \
  --test_data_list_path /data/bright/test.txt \
  --train_batch_size 4 \
  --crop_size 512 \
  --learning_rate 1e-4
```

ICR (adds instance consistency after warmup):
```bash
python train_MambaBDA_bright_ICR.py \
  --cfg /path/to/vssm_base_224.yaml \
  --pretrained_weight_path /path/to/pretrained.pth \
  --dataset BRIGHT \
  --train_dataset_path /data/bright/train \
  --val_dataset_path /data/bright/val \
  --test_dataset_path /data/bright/test \
  --train_data_list_path /data/bright/train.txt \
  --val_data_list_path /data/bright/val.txt \
  --test_data_list_path /data/bright/test.txt \
  --train_batch_size 4 \
  --crop_size 512 \
  --learning_rate 1e-4 \
  --inst_start_iter 1000 \
  --alpha 0.5 --beta 0.5
```

Force-directed (compactness + separation on HR features):
```bash
python train_MambaBDA_bright_ForceDirected.py \
  --cfg /path/to/vssm_base_224.yaml \
  --pretrained_weight_path /path/to/pretrained.pth \
  --dataset BRIGHT \
  --train_dataset_path /data/bright/train \
  --val_dataset_path /data/bright/val \
  --test_dataset_path /data/bright/test \
  --train_data_list_path /data/bright/train.txt \
  --val_data_list_path /data/bright/val.txt \
  --test_data_list_path /data/bright/test.txt \
  --train_batch_size 4 \
  --crop_size 512 \
  --learning_rate 1e-4 \
  --alpha 0.5 --beta 0.5
```

## Inference and evaluation
Use your trained checkpoint (e.g., `best_model.pth`) and run:
```bash
python infer_MambaBDA_BRIGHT.py \
  --cfg /path/to/vssm_base_224.yaml \
  --pretrained_weight_path /path/to/pretrained.pth \
  --dataset BRIGHT \
  --test_dataset_path /data/bright/test \
  --test_data_list_path /data/bright/test.txt \
  --result_saved_path /data/bright/results \
  --model_type MMMambaBDA \
  --resume /path/to/best_model.pth
```
Outputs:
- Localization masks under `result_saved_path/<dataset>/<model_type>/building_localization_map/`.
- Color-mapped damage predictions under `result_saved_path/<dataset>/<model_type>/damage_classification_map/`.
- Metrics printed for localization and damage per event and overall.

## Modules this depends on
- Configs: `changedetection/configs/config.py` + YAMLs consumed by `get_config`.
- Datasets: `changedetection/datasets/make_data_loader.py` providing `MultimodalDamageAssessmentDatset` and list-based loaders.
- Model: `changedetection/models/ChangeMambaMMBDA.py` (localization + damage heads over VMamba backbone).
- Losses/metrics: `changedetection/utils_func/` (Lovasz loss, evaluators, instance consistency losses for ICR/force-directed variants).
- Scripts: these entrypoints wire args → dataloaders → model → training/inference loops with AMP and checkpoints.
