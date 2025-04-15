# How to Set up your Environment

### Conda Installation

Anaconda is a Python based environment manager. It is one of most ubiquitous and standard methods for organizing and installing Python packages and works on many different Operating Systems. 

One great advantage of Conda for this project is that it comes with a set of channels specifically for bioinformatics tools - bioconda. With Conda & Bioconda we'll have easy access to all the tools we'll need to review, explore and model bacterial genomes.

I recommend using MiniConda (installs only the minimal functionality)

Installation details: https://docs.anaconda.com/free/miniconda/miniconda-install/
- The instructions will vary depending on OS (e.g. Mac OS vs Windows)
- Please follow the instructions at the link above to complete setting up MiniConda
- This step can be skipped if you already have Conda installed and set up on your machine

If you're comfortable doing so you can also install directly from the command line using:
https://docs.anaconda.com/free/miniconda/#quick-command-line-install 

##### A Note on MicroMamba

If performance and speed is super important an alternative option is to use MicroMamba which is written in C++ and can resolve and install packages much faster than Conda. 

Details: https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html

**NOTE** We will not be using or supporting MicroMamba during this project but feel free to look into it for your future projects.

### Bioconda

**NOTE** Bioconda is only supported on Mac OS or Linux. If you're using a Windows machine then you will need to set up a Linux virtual environment to be able to run Bioconda packages. This is only required for the initial data processing steps in Workshop 2. Already processed data will be provided in Workshop 3 and so if it isn't possible to run data bioconda on your machine we can try pairing up during Workshop 2 and this installation step can be safely skipped.

With Conda installed and set up BioConda is easy to set up and configure with the following commands:
```
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
conda config --set channel_priority strict
```

The above commands are enabling additional channels within conda which can then be searched for relevant packages to install.

Link to details: https://bioconda.github.io/ 

### Virtual Environment Setup

#### 1. Try using the conda env file (etbg_env.yml):
With Conda set up and configured, installed the required packages for this course is (hopefully!) straightforward. I've provided a yaml file with versions of each package we'll be using throughout the whole course. This is in the main root directory of the course folders and you can create your environment using:
`conda env create -f etbg_env.yml`

As biopython doesn't work on all platforms, if the above doesn't install you can try the version without any of the biopython packages:
`conda env create -f etbg_env_nobio.yml`

#### If automatic setup fails using the above YAML files, here is how you can create the environment yourself from scratch:
Run (type `y` if asked to proceed at each step):
```
conda create --name etbg-env python=3.10
conda activate etbg-env
conda install matplotlib
conda install seaborn
conda install ipykernel
conda install -c conda-forge jupyterlab
conda install scikit-learn
conda install tensorflow
```

Optional Biopython packages (Try but may fail on Windows):
```
conda install bowtie2
conda install -c conda-forge biopython
conda install kmer-jellyfish
conda install samtools
```

Please reach out on Slack and let me know if you cannot install the Biopython Packages. They are optional and we can work around it if they won't install on your system.

#### Test your environment:
(Make sure you're still in your conda environment, if not run: `conda activate etbg-env`)

To check if your environment is working run (from your command line):
```
ipython
from sklearn.metrics import accuracy_score
accuracy_score([0,0], [0,1])
```

You should see it return `0.5`

If so your environment is successfully set up!

Run (to leave the interactive python session):
```
exit
```



### Optional - Install an IDE

If you don't already have one on your machine, it is recommended to install an integrated development environment. This will make it much easier to organize your files, allows you to install helpful extensions (such as autoformatting) and enables you to run code all from a single place.

During this course the project will be demonstrated using VSCODE (and/or Jupyter Lab):
- Install instructions: https://code.visualstudio.com/

Useful extensions:
- Python
- Rainbow CSV
- Markdown Preview Enhanced
- Jupyter
