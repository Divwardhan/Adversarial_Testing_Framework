# GAN Experiments

A minimal adversarial training framework for medical image segmentation.

## Project Root

All commands should be run from the repository root:

```bash
cd /home/divwardhan/Desktop/Coding/GAN_experiments/main_experiments
```

## Setup

This project uses the local `uv` environment configured in the repository.

1. Install dependencies via `uv` if not already done:

```bash
uv sync
```

2. Run Python scripts using the `uv` environment:

```bash
uv run python main.py --help
```

## Data directory layout

The dataset should follow the structure expected by the project. For the glioma segmentation dataset, the path used in examples is:

```bash
./glioma/DATASET/Segmentation/Glioma
```

The repository also contains other dataset folders:

- `glioma/DATASET/classification/Training/...`
- `glioma/DATASET/classification/Testing/...`
- `glioma/DATASET/Segmentation/Glioma`
- `glioma/DATASET/Segmentation/Meningioma`
- `glioma/DATASET/Segmentation/Pituitary tumor`

`dataset/loader.py` expects paired image files and mask files, where each image is a PNG and its mask is named with `_mask.png` appended.

## Directory structure

- `main.py` - CLI entrypoint, parses arguments, starts training.
- `framework.py` - orchestrates the training workflow:
  - dataset loading
  - model/generator initialization
  - pretraining on clean data
  - alternating adversarial cycles
- `dataset/loader.py` - contains `TumorDataset` and `DatasetLoader`.
- `main_model/` - main segmentation model code:
  - `unet.py` - UNet architecture
  - `model.py` - `init_model()` factory
  - `train.py` - training functions for clean and adversarial training
- `generator/` - generator training and model code
- `utils/save.py` - saving adversarial sample outputs

## Code flow

1. `main.py` parses CLI options and calls `FrameworkRun(...)`.
2. `framework.py` loads the dataset using `DatasetLoader`.
3. `framework.py` initializes the segmentation model and adversarial generator.
4. `train_model_clean()` pretrains the model on clean data.
5. The framework runs alternating cycles:
   - generator training (`train_generator`)
   - model adversarial training (`train_model_adv`)
6. Adversarial samples are optionally saved under `outputs/adv_samples`.

## Run command

Use this command from the repository root:

```bash
uv run main.py --dataset_path ./glioma/DATASET/Segmentation/Glioma --device cuda --batch_size 32 --save_images --model_epochs 15 --pretrain_epochs 50 --lr_model 1e-4
```

If CUDA is unavailable, the script will fall back to CPU automatically.

## Notes

- `--save_images` stores adversarial examples in the `outputs/` folder.
- `--max_buffer_size` controls how many adversarial samples are kept during training.
- Use `--img_size` if you need a different input resolution.

Enjoy training and experimenting with the project!