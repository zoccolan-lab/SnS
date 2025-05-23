
# Installation

This project requires Python 3.10 or higher. We recommend using a virtual environment to manage dependencies.

## 1. Clone the repository (if you haven't already):

```bash
git clone <snslib-repository-url>
cd snslib
```

For the reviewing process, we provide a zip file containing the code base and the link to a Zenodo anonymized folder containing the useful data to easily run the code (in particular the demo section contains a Jupyter notebook to perform the exact analyses as in the paper).

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

Alternatively, using conda:

```bash
conda create -n snslib python=3.10
conda activate snslib
```

## 3. Install dependencies:

This project uses `pip` for package management. You can install the project along with its dependencies directly from the `pyproject.toml` file. This is the recommended way if you plan to use the `snslib` package itself:

```bash
cd /path/where/pyproject.toml/is/located
pip install -e .
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
*   If you encounter any issues during installation, please double-check that you have Python 3.10+ and `pip` installed and updated.
*   NOTE: In robustness folder (`robustness/imagenet_models`), the import 'from torchvision.models.utils import load_state_dict_from_url' is not working. Please change all occurencies of this import with 'from torch.hub import load_state_dict_from_url' instead.



# First steps: Local settings and Args:
In order to start performing your own experiments (MEI generation, Natural recordings,SnS experiments and also metaexperiments), you should first set the local settings and check the args.

## Local settings:

The local settings are stored in the `src/snslib/experiment/local_settings.json` file. They refer to the main files used in the experiments. In particular, you should specify the following fields:
1. `out_dir`: the path to the output folder where the results of your experiments will be saved. 
2. `weights`: the path to the folder containing the weights of the image generator. The fc7 variant is provided on Zenodo in the folder deepsim.
3. `dataset`: the path to the folder containing the mini-imagenet dataset. It can be downloaded from [here](https://www.kaggle.com/datasets/arjunashok33/miniimagenet).
   
   **NOTE**: Remember to add the `inet_labels.txt` file in the mini-imagenet dataset folder from [here](https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57#file-map_clsloc-txt)
4. `references`: the path to the file containing the unified reference.pkl file. If you don't have one yet, fill it in with the path to the folder where you want the reference.pkl file to be stored. We uploaded a unified_references.pkl file on Zenodo.
5. `custom_weights`: the path to the folder containing the weights of the robust version of the subject network. It can be downloaded from the robustness library [here](https://www.dropbox.com/s/knf4uimlqsi1yz8/imagenet_l2_3_0.pt?dl=0). This should be in a folder called `resnet50` and the path should be the path to the folder containing the `resnet50` folder.
6. `natural_recordings`: the path to the file containing the unified natural recordings.pkl file. If you don't have one yet, fill it in with the path to the folder where you want the natural recordings.pkl file to be stored. We uploaded a unified_nat_recs.pkl file on Zenodo.
   
For details on how to update references and natural recordings, please refer to the following sections.

## Args:

Args are the parameters used in the experiments. Every experiment has its own args. A detailed description of the args can be found in the `src/snslib/experiment/utils/args.py` file, to which the args of the single experiment refer to.

# How to generate MEIs (Most Exciting Images) as reference for SnS experiments:


Fill in the `src/snslib/experiment/local_settings.json` file with the correct paths to the weights (specify the path to the folder which contains the 'fc7.pt' file).


In the `src/experiments/MaximizeActivity/run` directory you can find the `multirun_arguments.py` file. This file contains the parameters for multiple MEIs generation. Set the parameters following the comments along the script and run the script to get the prompt for generating the MEIs. The prompt will be automatically saved in the `src\experiments\MaximizeActivity\run\multirun_cmd.txt` file. Copy it and paste it on terminal:

```bash
cd src/experiments/MaximizeActivity/run
your_prompt_file
```
Output data will be saved as `data.pkl` file in `out_dir` folder.

Now, set the path to your reference file (.pkl file) in the `src/snslib/experiment/local_settings.json` file. If you already have a reference file, please indicate its path. If not, just indicate the path where you want the reference.pkl file to be stored.


 To unify the references from different runs, you should run the `unify_references.ipynb` notebook.




# How to record activity from natural images:

For the early stopping it is vital to record the activity of the target neurons when they are stimulated by natural images. In particular the training set of Imagenet, or the mini-imagenet dataset.
To do so, you should follow these steps:


0. Set the appropriate params in the `src\experiments\NeuralRecording\args.py` file;

**NOTE**: In `ExperimentArgParams.Dataset.value` set the path to either mini-imagenet dataset (which you should already have downloaded for the MEI generation) or Imagenet dataset (the latter was used in the paper's experiments, to download it go to [here](https://huggingface.co/datasets/ILSVRC/imagenet-1k)).
1. Run the `src\experiments\NeuralRecording\run\run_single.py` file to record the activity of the target neurons.

As for references, neural recordings data from multiple runs should be unified in a single file, to do this, you should:

2. Fill in the `src\snslib\experiment\local_settings.json` file with the paths to the unified natural recordings file (if you already have one, please indicate its path. If not, just indicate the path where you want the unified natural recordings file to be stored);
3. Fill in the `src\experiments\NeuralRecording\nat_stats_fp.json` file with the paths to the neural recordings data from multiple runs;
4. Run the `src\experiments\NeuralRecording\nat_stats_dict.ipynb` file to unify the neural recordings data from multiple runs.
5. Add the path to the unified neural recordings data in the `src/snslib/experiment/local_settings.json` file.

# How to run a single SnS experiment:

Set the parameters as you want in the `src\experiments\Stretch_and_Squeeze\args.py` file.If you want to run a SnS experiment for an Adversarial task:

Run the `src\experiments\Stretch_and_Squeeze\run\run_single.py` file.

If you want to run a SnS experiment for Invariance task:

Run the `src\experiments\Stretch_and_Squeeze\run\run_single_ri.py` file.

# How to run multiple SnS experiments:

In the `src\experiments\Stretch_and_Squeeze\run\multirun_arguments.py` file set the path to save subsampling spaces (SPACE_TXT = 'path/to/save/subsampling/low_spaces.txt', in the top
of the script).

Run the `src\experiments\Stretch_and_Squeeze\run\multirun_arguments.py` file to get the prompt for running the multiple SnS experiments.The prompt will be automatically saved in the `src\experiments\Stretch_and_Squeeze\run\multirun_cmd.txt` file.
Once you copy the prompt, you can run the multiple SnS experiments with the following command:

```bash
cd src/experiments/Stretch_and_Squeeze/run
your_prompt_file
```

Note: this section can be used for running the subsampling experiment as well (see comments in the `multirun_arguments.py` file).


# Metaexperiments

Once you have your SnS experiments results, you can run the metaexperiments.
A metaexperiment is a collection of multiple SnS experiments that allows analyses between them. 
To run a metaexperiment, you should follow these steps:

0. set the path to `hyperparams_meta_an.json` file in the `src\snslib\metaexperiment\metaexp.py` file (i.e. the variable HYPERPARAMS_FP at the top of the file) and fill in `hyperparams_meta_an.json` with the required information (NOTE: in nat_rec_fp you should indicate the path to the unified natural recordings file);
1. Fill in the `src\snslib\metaexperiment\SnS_multiexp_dirs.json` file with the paths to the SnS experiments results; We provide the SnS experiments used in the paper in the Zenodo repository.
2. Add to `src\snslib\metaexperiment\metaexp_functs.py` the path to the Full_inet_labels.npy file provided in the Zenodo repository. 
3. After filling in the required paths, run the `demo/paper_analysis.ipynb` to perform the analyses as in the paper.

**NOTE**: two .json files are important for running analysis in the `demo` folder:
1) `distance_params.json`: contains the parameters for the distance analysis. You should set: 
   - your analysis name exp_name;
   - your observer net ref_net (set net name following pytorch nomenclature, e.g. resnet50);
   - Fill in the "robust" with 'imagenet_l2_3_0.pt' if you want to use the robust resnet50, or '' if you want to use the non-robust resnet50;
   - Fill in the "gen" with 'fc7' if you want to use the MEI reference;
   - Fill in the "SNS_exp" with the fields of the queries for  all the multiexps you want to analyze.
   - The XDREAM section is used to get the XDREAM references. In these subfields speficify the path to unified_references.pkl file (i.e. the same of local_settings.json) in the field 'fp.
   - The "plotting" section is used to plot the results with the desired parameters (color, linestyle, label, etc.). Note that the keys of this section should be the same as the keys of the SnS experiments in the "queries" section but with the "#" symbol separating the different fields of the query (e.g. "resnet50#56_linear_01#robust_l2#00_input_01#500#invariance 2.0#VSref").
2) `hyperparams_meta_an.json`: contains the parameters for the Centroid analysis. You should set:
   - The savepath field with the path to the folder where you want to save the plots;
   - The plotting field with the parameters for the plotting of the results (e.g. color, marker, label, etc.).
   - The "nat_percentiles" field is used to get the natural_recordings.pkl, it should point to the path you set in the local_settings.json file.
   - The "SnS_scatterplot" field is used to get the scatterplot of the SnS experiments, plot_type should be set to "centroid" and end_type should be set to "end" to repeat the analysis of the paper.

# Additional notes:
The whole code base was tested on a Linux machine (Ubuntu 22.04) and a Nvidia GTX 1080Ti GPU.


