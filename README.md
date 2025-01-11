# TVLARS - Pytorch Official Implementation

# Abstract
<p align="justify">LARS and LAMB have emerged as prominent techniques in Large Batch Learning (LBL), ensuring the stability of AI training. One of the primary challenges in LBL is stabilizing convergence, where the AI agent is usually get trapped in the sharp minimizer. Addressing this challenge, a relatively recent technique, known as warmup, has been employed. However, it's worth noting that warmup lacks a strong theoretical foundation, leaving the door open for further exploration of more efficacious algorithms. In light of this situation, we conducted empirical experiments to analyze the behaviors of the two most popular optimizers in the LARS family: LARS and LAMB, with and without a warm-up strategy. Our analysis gives us a comprehension of the novel LARS, LAMB, and the necessity of a warmup technique in LBL. Building upon these insights, we propose a novel algorithm called Time Varying LARS (TVLARS), which facilitates robust training in the initial phase without the need for warm-up. Experimental evaluation demonstrates that TVLARS achieves competitive results with LARS and LAMB when warm-up is utilized while surpassing their performance without the warm-up technique.</p>

# Experiment
## Setup
This work can be conducted on any platform: Windows, Ubuntu, Google Colab. In Windows or Ubuntu use the following script to create a virtual environment.
```
git clone https://github.com/KhoiDOO/tvlars.git
cd path/to/tvlars
python -m venv .env
```
The Python packages used in this project are listed below. Crucially, ```parquet``` and ```pyarrow``` are used for writing and saving ```.parquet``` file, which is a strongly compressed file for saving the DataFrame. All the packages can be installed by command ```pip install -r requirements.txt```. If ```parquet``` does not work with your machine, consider using ```fastparquet``` instead.
```
matplotlib==3.7.1
numpy==1.24.3
pandas==2.0.1
parquet==1.3.1
pyarrow==12.0.0
seaborn==0.12.2
tqdm==4.65.0
```
[Pytorch](https://pytorch.org/) is the main package for conducting optimization calculations, whose version is ```2.0.1```.
## Available Settings
Using ```python main.py -h``` to print out all available settings of this project. The table below show the tag as well as its related description. 
| **TAG**                       | **OPTIONS**                                  | **DESCRIPTION**                                                               |
|-------------------------------|----------------------------------------------|-------------------------------------------------------------------------------|
| -h, --help                    |                                              | show this help message and exit                                               |
| --bs BS                       |                                              | batch size                                                                    |
| --workers WORKERS             |                                              | Number of processor used in data loader                                       |
| --epochs EPOCHS               |                                              | Number of epochs used in training                                             |
| --lr LR                       |                                              | initial learning rate                                                         |
| --seed SEED                   |                                              | seed for initializing training                                                |
| --port PORT                   |                                              | Multi-GPU Training Port                                                       |
| --wd W                        |                                              | weight decay                                                                  |
| --ds                          | cifar10, cifar100, tinyimagenet              | data set name                                                                 |
| --model                       | resnet18, resnet34, resnet50, effb0          | model used in training                                                        |
| --opt                         | adam, adamw, adagrad, rmsprop, lars, tvlars, lamb | optimizer used in training                                               |
| --sd                          | None, cosine, lars-warm                       | learning rate scheduler used in training                                      |
| --dv DV [DV ...]              |                                              | list of devices used in training                                              |
| --lmbda LMBDA                 |                                              | delay factor used in TVLARS                                                   |
| --cl_epochs CL_EPOCHS         |                                              | epoch used in Barlow twins feature redundant removal stage                    |
| --btlmbda BTLMBDA             |                                              | lambda factor used in Barlow Twins                                            |
| --projector PROJECTOR         |                                              | dimensions of top Multilayer Perceptron used in Barlow Twins                  |
| --lr_classifier LR_CLASSIFIER |                                              | classifier learning rate used in Barlow Twins                                 |
| --lr_backbone LR_BACKBONE     |                                              | backbone learning rate used in Barlow Twins                                   |
| --mode                        | clf, bt                                      | experiment mode, clf is for classification, bt is for Barlow Twins experiment |

## Running
For instance, the experiment of TVLARS with batch size ($\mathcal{B}$) of ```512``` and various delay factor ($\lambda$) values by the following expressions:
**Classification Experiment**
```
python main.py --bs 512 --epochs 100 --lr 1.0 --port 7046 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 1e-06 --dv 0 1 2 3
python main.py --bs 512 --epochs 100 --lr 1.0 --port 3675 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 1e-05 --dv 0 1 2 3
python main.py --bs 512 --epochs 100 --lr 1.0 --port 6162 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 0.0001 --dv 0 1 2 3
python main.py --bs 512 --epochs 100 --lr 1.0 --port 3930 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 0.001 --dv 0 1 2 3
python main.py --bs 512 --epochs 100 --lr 1.0 --port 7644 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 0.005 --dv 0 1 2 3
python main.py --bs 512 --epochs 100 --lr 1.0 --port 5794 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 0.01 --dv 0 1 2 3
python main.py --bs 512 --epochs 100 --lr 2.0 --port 3976 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 1e-06 --dv 0 1 2 3
python main.py --bs 512 --epochs 100 --lr 2.0 --port 5895 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 1e-05 --dv 0 1 2 3
python main.py --bs 512 --epochs 100 --lr 2.0 --port 5014 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 0.0001 --dv 0 1 2 3
python main.py --bs 512 --epochs 100 --lr 2.0 --port 6423 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 0.001 --dv 0 1 2 3
python main.py --bs 512 --epochs 100 --lr 2.0 --port 5228 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 0.005 --dv 0 1 2 3
python main.py --bs 512 --epochs 100 --lr 2.0 --port 6169 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 0.01 --dv 0 1 2 3
python main.py --bs 512 --epochs 100 --lr 3.0 --port 5466 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 1e-06 --dv 0 1 2 3
python main.py --bs 512 --epochs 100 --lr 3.0 --port 7422 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 1e-05 --dv 0 1 2 3
python main.py --bs 512 --epochs 100 --lr 3.0 --port 6373 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 0.0001 --dv 0 1 2 3
python main.py --bs 512 --epochs 100 --lr 3.0 --port 6592 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 0.001 --dv 0 1 2 3
python main.py --bs 512 --epochs 100 --lr 3.0 --port 4802 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 0.005 --dv 0 1 2 3
python main.py --bs 512 --epochs 100 --lr 3.0 --port 7327 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 0.01 --dv 0 1 2 3
```
**Barlow Twins Experiment**
```
python main.py --bs 512 --epochs 100 --cl_epochs 1000 --lr 1.0 --port 7186 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 1e-06 --dv 0 1 2 3 --mode bt
python main.py --bs 512 --epochs 100 --cl_epochs 1000 --lr 1.0 --port 4111 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 1e-05 --dv 0 1 2 3 --mode bt
python main.py --bs 512 --epochs 100 --cl_epochs 1000 --lr 1.0 --port 4356 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 0.0001 --dv 0 1 2 3 --mode bt
python main.py --bs 512 --epochs 100 --cl_epochs 1000 --lr 1.0 --port 7782 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 0.001 --dv 0 1 2 3 --mode bt
python main.py --bs 512 --epochs 100 --cl_epochs 1000 --lr 1.0 --port 4353 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 0.005 --dv 0 1 2 3 --mode bt
python main.py --bs 512 --epochs 100 --cl_epochs 1000 --lr 1.0 --port 6524 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 0.01 --dv 0 1 2 3 --mode bt
python main.py --bs 512 --epochs 100 --cl_epochs 1000 --lr 2.0 --port 3979 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 1e-06 --dv 0 1 2 3 --mode bt
python main.py --bs 512 --epochs 100 --cl_epochs 1000 --lr 2.0 --port 4969 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 1e-05 --dv 0 1 2 3 --mode bt
python main.py --bs 512 --epochs 100 --cl_epochs 1000 --lr 2.0 --port 3517 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 0.0001 --dv 0 1 2 3 --mode bt
python main.py --bs 512 --epochs 100 --cl_epochs 1000 --lr 2.0 --port 7895 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 0.001 --dv 0 1 2 3 --mode bt
python main.py --bs 512 --epochs 100 --cl_epochs 1000 --lr 2.0 --port 4434 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 0.005 --dv 0 1 2 3 --mode bt
python main.py --bs 512 --epochs 100 --cl_epochs 1000 --lr 2.0 --port 7770 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 0.01 --dv 0 1 2 3 --mode bt
python main.py --bs 512 --epochs 100 --cl_epochs 1000 --lr 3.0 --port 5348 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 1e-06 --dv 0 1 2 3 --mode bt
python main.py --bs 512 --epochs 100 --cl_epochs 1000 --lr 3.0 --port 4362 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 1e-05 --dv 0 1 2 3 --mode bt
python main.py --bs 512 --epochs 100 --cl_epochs 1000 --lr 3.0 --port 6193 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 0.0001 --dv 0 1 2 3 --mode bt
python main.py --bs 512 --epochs 100 --cl_epochs 1000 --lr 3.0 --port 6442 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 0.001 --dv 0 1 2 3 --mode bt
python main.py --bs 512 --epochs 100 --cl_epochs 1000 --lr 3.0 --port 7169 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 0.005 --dv 0 1 2 3 --mode bt
python main.py --bs 512 --epochs 100 --cl_epochs 1000 --lr 3.0 --port 7954 --wd 0.0005 --ds cifar10 --model resnet18 --opt tvlars --sd None --lmbda 0.01 --dv 0 1 2 3 --mode bt
```

# Citation
```
@ARTICLE{tvlars,
      author={Do, Khoi and Nguyen, Minh-Duong and Hoa, Nguyen Tien and Tran-Thanh, Long and Tran, Nguyen H. and Pham, Quoc-Viet},
      journal={IEEE Transactions on Artificial Intelligence}, 
      title={Revisiting LARS for Large Batch Training Generalization of Neural Networks}, 
      year={2024},
      volume={},
      number={},
      pages={1-12},
      keywords={Training;Convergence;Artificial intelligence;Vectors;Computer science;Accuracy;Optimization;Hardware;Faces;Eigenvalues and eigenfunctions;Artificial intelligence algorithmic design and analysis;Classification and regression;Deep learning;Machine learning},
      doi={10.1109/TAI.2024.3523252}
}
```
