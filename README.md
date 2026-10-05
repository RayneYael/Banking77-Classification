# Banking77 BERT Baseline

This repository currently contains an end-to-end baseline for intent classification on
[Banking77](https://huggingface.co/datasets/PolyAI/banking77). The pipeline in
`baseline.ipynb` covers data loading, a stratified train/validation split, tokenization,
BERT fine-tuning, evaluation, and error-analysis visualizations.

This is a working baseline and a progress update, not the final project submission.

> **Want to skip training?** Jump directly to
> [Option B: evaluate the cached best model](#option-b-skip-training-and-evaluate-the-cached-best-model).
> This is the optional quick-start path and only requires running selected notebook cells.

## Files

- `baseline.ipynb`: the complete training and evaluation pipeline.
- `best_bert_mlp.pt`: the best checkpoint selected by validation macro-F1.
- `requirements-baseline.txt`: pinned Python dependencies from the environment used to
  run the notebook.

The checkpoint is a PyTorch `state_dict` for `BertForSequenceClassification` with a
`bert-base-uncased` backbone and 77 output labels. Despite the current filename, the
notebook does not add a separate custom MLP head.

## Environment setup

The saved notebook run used the following environment:

- Linux/WSL2
- Python 3.14.6
- PyTorch 2.14.1+cu126
- CUDA was selected by the notebook for the saved training run

Create an isolated environment and install the dependencies:

```bash
conda create -n ca6001-baseline python=3.14.6 -y
conda activate ca6001-baseline

# Install PyTorch first. This reproduces the CUDA build used for the saved run.
pip install torch==2.14.1+cu126 --index-url https://download.pytorch.org/whl/cu126

# Install the remaining pinned dependencies.
pip install -r requirements-baseline.txt

python -m ipykernel install --user \
  --name ca6001-baseline \
  --display-name "Python (CA6001 baseline)"
```

If the machine does not have a compatible NVIDIA GPU, install the matching CPU build of
PyTorch instead. The notebook automatically falls back to CPU, although full fine-tuning
will be substantially slower. A different recent Python/PyTorch combination may work, but
the versions above are the versions used by the saved run.

On the first run, internet access is required to download:

- the `PolyAI/banking77` dataset;
- the `bert-base-uncased` tokenizer, configuration, and pretrained base weights.

Hugging Face caches these resources locally for later runs. The file
`best_bert_mlp.pt` contains only the fine-tuned model parameters; it does not contain the
dataset or tokenizer.

Start Jupyter after installation:

```bash
jupyter lab baseline.ipynb
```

Select the **Python (CA6001 baseline)** kernel before running any cells.

## Option A: run the complete pipeline

Use **Kernel > Restart Kernel and Run All Cells**. Running from top to bottom is important
because later cells reuse variables created earlier.

The training configuration recorded in the notebook is:

| Setting | Value |
| --- | ---: |
| Dataset | `PolyAI/banking77` |
| Base model | `bert-base-uncased` |
| Random seed | 42 |
| Validation split | 15% of the original training set, stratified by label |
| Maximum sequence length | 128 |
| Batch size | 32 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Epochs | 5 |
| Warmup ratio | 0.1 |
| Number of labels | 77 |

During training, the checkpoint is overwritten whenever validation macro-F1 improves.
The saved run reached validation macro-F1 **0.8924** at epoch 5.

## Option B: skip training and evaluate the cached best model

Cell numbers below are **zero-based positional indices**: the first cell in the notebook
is cell 0. Use the first comment shown below as an additional visual check.

1. Restart the kernel to remove stale variables.
2. Run cells **0 through 11**, in order. These cells perform imports, load and split the
   dataset, build the label mapping, initialize the tokenizer and DataLoaders, instantiate
   the 77-label BERT architecture, and define the evaluation function.
3. **Do not run cell 12** (`# Training`). This is the expensive fine-tuning loop and can
   overwrite `best_bert_mlp.pt`.
4. **Do not run cells 13 and 14** (`# Training curve: Loss` and
   `# Validation accuracy and macro-F1`). They depend on the `history` object created by
   the skipped training cell.
5. Run cell **15** (`# Load best model`). It loads `best_bert_mlp.pt` onto the selected
   CPU or GPU.
6. Run cells **16 through 19** for test metrics, the classification report, lowest-F1
   classes, normalized confusion matrix, and common confusion pairs.

In short:

```text
Run:  0–11  ->  15–19
Skip: 12–14
```

Keep `baseline.ipynb` and `best_bert_mlp.pt` in the same working directory. If the
notebook server was launched elsewhere, either start it from this repository directory or
change the checkpoint path in cell 15.

Do not change `MODEL_NAME`, `NUM_LABELS`, or the Banking77 label mapping when loading this
checkpoint. To reproduce the recorded evaluation values, also keep `SEED=42`,
`VALIDATION_RATIO=0.15`, and `MAX_LENGTH=128` unchanged.

Checkpoint identity:

```text
File:   best_bert_mlp.pt
Size:   438,249,031 bytes (about 418 MiB)
SHA256: 7187a98159d123d3d11742afbd98738b11d45adbdf9c5a71f41579383d8a2ab7
```

You can verify it with:

```bash
sha256sum best_bert_mlp.pt
```

## Expected cached-model results

With the saved split and configuration, cell 16 should report approximately:

| Metric | Value |
| --- | ---: |
| Test loss | 0.6293 |
| Test accuracy | 0.9003 |
| Macro precision | 0.9087 |
| Macro recall | 0.9003 |
| Macro F1 | 0.8982 |

Small numerical differences can occur across hardware and library builds. Large differences
usually indicate a different dataset split, label order, preprocessing configuration, or
checkpoint file.

## Notes and known limitations

- `MLP_HIDDEN_SIZE` and `DROPOUT` are currently declared in the hyperparameter cell but
  are not used by the model construction code.
- The checkpoint is about 418 MiB, which exceeds GitHub's normal 100 MiB per-file limit.
  If the repository is shared through GitHub, distribute it through Git LFS or a separate
  shared download location and retain the filename and checksum above.
- The notebook's saved dataset output refers to Banking77 dataset configuration version
  `1.1.0`. Keeping the pinned `datasets` version and the same dataset source is recommended
  for reproduction.
