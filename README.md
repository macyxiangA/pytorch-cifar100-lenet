# PyTorch CIFAR-100 & Lightweight Language Model Experiments

This repository contains a collection of lightweight PyTorch experiments focused on:
- image classification on CIFAR-100 using a LeNet-style CNN
- small-scale language modeling and text generation experiments designed to run on CPU

The codebase is organized as a compact research / experimentation workspace rather than a single monolithic project.

## Repository Structure

```text
.
|-- __pycache__/                 # Python cache files
|-- data/                        # CIFAR-100 data directory (auto-created)
|-- distilgpt2_sft_cpu/          # Saved small language model artifacts
|-- nanoGPT/                     # nanoGPT-based experiments
|-- outputs/                     # Model checkpoints and outputs
|-- data.csv                     # Small text dataset for language modeling
|-- dataloader.py                # CIFAR-100 download/extract/cache + Dataset class
|-- eval_cifar100.py             # Evaluate CIFAR-100 classification model
|-- generated_distilgpt.txt      # Sample generated text output
|-- make_data_csv.py             # Build a small CSV dataset from WikiText-2
|-- README.md                    # Project documentation
|-- result.txt                   # Recorded experiment results
|-- student_code.py              # LeNet model + training/testing helpers
|-- train_cifar100.py            # CIFAR-100 training entrypoint
```

## CIFAR-100 Image Classification

```text
Core files:
- dataloader.py
- student_code.py
- train_cifar100.py
- eval_cifar100.py
```

### Model

```text
Architecture: LeNet-style CNN
Input: 32x32 RGB image

Conv(3 -> 6, kernel=5) + ReLU + MaxPool
Conv(6 -> 16, kernel=5) + ReLU + MaxPool
Flatten: 16 * 5 * 5 = 400
FC: 400 -> 256 -> 128 -> 100

Loss: CrossEntropyLoss
Optimizer: SGD with momentum
```

The forward pass can optionally return intermediate tensor shapes to assist with debugging.

### Training

```text
python train_cifar100.py --epochs 10 --batch-size 32 --lr 0.001
```

```text
Arguments:
--epochs        number of training epochs
--batch-size    mini-batch size
--lr            learning rate
--resume        path to a checkpoint to resume from
```

```text
Checkpoints are saved to:
./outputs/
```

### Evaluation

```text
python eval_cifar100.py
```

```text
Optional:
python eval_cifar100.py --load outputs/model_best.pth.tar
```

Evaluation reports classification accuracy on the CIFAR-100 test set.

## CIFAR-100 Dataset Handling

```text
Behavior:
- Automatically downloads the CIFAR-100 archive if missing
- Verifies integrity using MD5
- Extracts and converts raw data into image files
- Caches processed samples for faster reloads
```

No manual dataset preparation is required.

## Lightweight Language Modeling Experiments

```text
Relevant files:
- make_data_csv.py
- data.csv
- distilgpt2_sft_cpu/
- nanoGPT/
- generated_distilgpt.txt
```

```text
make_data_csv.py:
- Loads WikiText-2 (raw)
- Filters empty lines
- Builds a small single-column CSV ("text")
- Designed for fast CPU-based experimentation
```

```text
distilgpt2_sft_cpu/:
- Stores tokenizer files
- Stores model weights and configs
- Used for small-scale supervised fine-tuning and generation
```

```text
nanoGPT/:
- Experimental sandbox for character-level or small-scale GPT training
```

## Environment Requirements

```text
Python 3.9+
torch
torchvision
tqdm
numpy
pillow
datasets
pandas
```

## Notes

```text
- This repository is intended for experimentation and learning.
- Scripts are designed to run on CPU with modest resource requirements.
- Some directories (data/, outputs/) are created automatically at runtime.
```

## License

```text
MIT
```
