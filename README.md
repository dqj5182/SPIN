# SPIN - SMPL oPtimization IN the loop
Code repository for the paper:  
**Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop**  
[Nikos Kolotouros](https://www.nikoskolot.com/)\*, [Georgios Pavlakos](https://geopavlakos.github.io/)\*, [Michael J. Black](https://ps.is.mpg.de/~black), [Kostas Daniilidis](http://www.cis.upenn.edu/~kostas/)  
ICCV 2019  
[[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kolotouros_Learning_to_Reconstruct_3D_Human_Pose_and_Shape_via_Model-Fitting_ICCV_2019_paper.pdf)] [[project page](https://www.nikoskolot.com/projects/spin/)]

![teaser](assets/teaser.png)


## Installation instructions
Start locally:
```
git clone https://github.com/dqj5182/SPIN.git
cd SPIN
```

Create conda environment and install PyTorch and other packages:
```
# Initialze conda env
conda create -n spin python=3.9
conda activate spin

# Install PyTorch and other packages
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt

pip install tensorrt

# Install face detection
pip install deepface
pip install facenet-pytorch
```

* If you encounter error like "RuntimeError: The detected CUDA version (11.6) mismatches the version that was used to compile PyTorch (10.2). Please make sure to use the same CUDA versions.":
```
export PATH=/usr/local/cuda-10.2/bin:/usr/local/cuda-10.2/NsightCompute-2019.1${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64\ ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
or refer to [reference](https://stackoverflow.com/questions/53422407/different-cuda-versions-shown-by-nvcc-and-nvidia-smi).

## Download data
This provides necessary data for training and evaluation. Please run:
```
./fetch_data.sh
```

Please also download SMPL human model files from [Google Drive](https://drive.google.com/drive/folders/1A7c0LsiHo4vznIajx3lCfLGEKHEXDEcD?usp=sharing) and move the files under `data`. </br>
Note that these files under license.

## Data
You need to follow directory structure of the `data` as below.
```
${ROOT} 
|-- data  
|   |-- dataset_extras
|   |   |--3dpw_test.npz
|   |   |--coco_2014_train.npz
|   |   |--h36m_valid_protocol1.npz
|   |   |--h36m_valid_protocol2.npz
|   |   |--hr-lspet_train.npz
|   |   |--lsp_dataset_original_train.npz
|   |   |--lsp_dataset_test.npz
|   |   |--mpi_inf_3dhp_train.npz
|   |   |--mpi_inf_3dhp_valid.npy
|   |   |--mpii_train.npz
|   |-- smpl
|   |   |--SMPL_FEMALE.pkl
|   |   |--SMPL_MALE.pkl
|   |   |--SMPL_NEUTRAL.pkl
|   |-- static_fits
|   |   |--coco_fits.npy
|   |   |--lsp-orig_fits.npy
|   |   |--lspet_fits.npy
|   |   |--mpi-inf-3dhp_fits.npy
|   |   |--mpi-inf-3dhp_mview_fits.npz
|   |   |--mpii_fits.npy
|   |-- cube_parts.npy
|   |-- gmm_08.pkl
|   |-- J_regressor_extra.npy
|   |-- J_regressor_h36m.npy
|   |-- model_checkpoint.pt
|   |-- README.md
|   |-- smpl_mean_params.npz
|   |-- train.h5
|   |-- vertex_texture.npy
```

## Evaluation
Before evaluating SPIN model, please prepare 3DPW dataset. </br>
For preparing the dataset, please contact me personally!
</br>


Please run:
```
python eval.py --checkpoint=data/model_checkpoint.pt --dataset=3dpw --log_freq=20
```
The results should be:
```
MPJPE: 96.98920413126925
Reconstruction Error (PA-MPJPE): 59.41015338496593
```