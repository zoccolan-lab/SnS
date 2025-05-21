
# Installation

This project requires Python 3.9 or higher. We recommend using a virtual environment to manage dependencies.

## 1. Clone the repository (if you haven't already):

```bash
git clone <snslib-repository-url>
cd snslib
```

## 2. Create and activate a virtual environment:

*   **On macOS and Linux:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

*   **On Windows:**

    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

    (If you are using Git Bash on Windows, use `source venv/Scripts/activate`)

## 3. Install dependencies:

This project uses `pip` for package management. You can install the project along with its dependencies directly from the `pyproject.toml` file. This is the recommended way if you plan to use the `snslib` package itself:

```bash
pip install .
```

This command will install the `snslib` package and all its dependencies as defined in `pyproject.toml`.

## 4. Verify installation (Optional):

You can verify that the core libraries are installed by opening a Python interpreter and trying to import them:

```python
import torch
import torchvision
import numpy
import scipy
# ... and so on for other key dependencies
```

If no errors occur, the installation was successful.

---

**Notes:**

*   **PyTorch Installation:** The `torch` and `torchvision` packages can sometimes have specific installation needs depending on your operating system and whether you want CUDA support (for GPU acceleration). If you encounter issues, please refer to the official PyTorch installation guide: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
*   **robustbench:** This dependency is installed directly from a GitHub repository. Ensure you have `git` installed and accessible in your PATH for this to work correctly.
*   If you encounter any issues during installation, please double-check that you have Python 3.9+ and `pip` installed and updated.

# How to generate MEIs (Most Exciting Images) as reference for SnS experiments:

In the `src/experiments/MaximizeActivity/run` directory you can find the `multirun_arguments.py` file. This file contains the parameters for the MEI generation. Set the parameters as you want and run the script to get the prompt for generating the MEIs. The prompt will be automatically saved in the `src\experiments\MaximizeActivity\run\multirun_cmd.txt` file. Copy it and paste it on terminal:

```bash
cd src/experiments/MaximizeActivity/run
your_prompt_file
```
Output data will be saved as `data.pkl` file in `out_dir` folder. To unify the references from different runs, you should run the `unify_references.ipynb` notebook.

# How to record activity from natural images:

For the early stopping it is vital to record the activity of the target neurons when they are stimulated by natural images. In particular the training set of Imagenet.
To do so, you should follow these steps:

1. Download the Imagenet dataset (e.g. from [here](https://huggingface.co/datasets/ILSVRC/imagenet-1k));
2. Set the appropriate params in the `src\experiments\NeuralRecording\args.py` file;
3. Run the `src\experiments\NeuralRecording\run_single.py` file to record the activity of the target neurons.

As for references, neural recordings data from multiple runs should be unified in a single file, to do this, you should:
1. Fill in the `src\experiments\NeuralRecording\nat_stats_fp.json` file with the paths to the neural recordings data from multiple runs;
2. Run the `src\experiments\NeuralRecording\nat_stats_dict.ipynb` file to unify the neural recordings data from multiple runs.

# How to run a single SnS experiment:

Set the parameters as you want in the `src\experiments\Stretch_and_Squeeze\args.py` file.If you want to run a SnS experiment for an adversarial task:

Run the `src\experiments\Stretch_and_Squeeze\run\run_single.py` file.

If you want to run a SnS experiment for Invariance task:

Run the `src\experiments\Stretch_and_Squeeze\run\run_single_ri.py` file.

# How to run multiple SnS experiments:

Run the `src\experiments\Stretch_and_Squeeze\run\multirun_arguments.py` file to get the prompt for running the multiple SnS experiments.The prompt will be automatically saved in the `src\experiments\Stretch_and_Squeeze\run\multirun_cmd.txt` file.
Once you have the prompt, you can run the multiple SnS experiments with the following command:

```bash
cd src/experiments/Stretch_and_Squeeze/run
your_prompt_file
```

Note: this section can be used for running the subsampling experiment as well (see comments in the `multirun_arguments.py` file).


