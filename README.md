# pan-motion-retargeting
Official implementation for the paper ["Pose-aware Attention Network for Flexible Motion Retargeting by Body Part"](https://ieeexplore.ieee.org/document/10129844)

Please visit our [webpage](https://hlcdyy.github.io/pan-motion-retargeting/) for more details.

![hum2dog](https://raw.githubusercontent.com/hlcdyy/pan-motion-retargeting/pan-page/static/hum2dog.gif)

## Getting started 
This code was tested on `Ubuntu 18.04.4 LTS` and requires:
* Python 3.8
* conda3
* CUDA capable GPU 

### 1. Creat conda environment
<details><summary>Click to expand</summary>

We strongly recommend activating a Python virtual environment prior to installing PAN. Follow these steps to download and install it. Then run the following commands:
````
# create and activate the virtual environment
conda create --name pan_retargeting python=3.8.12
conda activate pan_retargeting 
````
Install [PyTorch 1.10.0](https://pytorch.org/) inside the conda environment.

````
# clone pan-motion-retargeting and use pip to install
git clone https://github.com/hlcdyy/pan-motion-retargeting.git
cd pan-motion-retargeting
pip install -e .
````
</details>

### 2. Download the datasets
<details><summary>Click to expand</summary>

**Mixamo dataset** 

**Be sure to read and follow their license agreements, and cite accordingly.**

We use [Mixamo](https://www.mixamo.com/#/) dataset to train our model for retargeting between humanoid characters and follow the train-test setting of [SAN](https://github.com/DeepMotionEditing/deep-motion-editing). You can find the download link in the [SAN github page](https://github.com/DeepMotionEditing/deep-motion-editing) for the preprocessed data or generating the data from scratch. 

The `Mixamo` directory should be placed within `data_preprocess/Mixamo`

**Lafan1 and Dog datasets**

**Be sure to read and follow their license agreements, and cite accordingly.**

Creat this folder:
````
mkdir data_preprocess/Lafan1_and_dog/Lafan1
````
Go to the [Lafan1 website](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) and download the lafan1.zip. Then unzip it and put all the .bvh files into `data_preprocess/Lafan1_and_dog/Lafan1`

Creat this folder:
````
mkdir data_preprocess/Lafan1_and_dog/DogSet
```` 
Go to the [AI4Animation Website](https://github.com/sebastianstarke/AI4Animation) and get the Mocap Data from "Mode-Adaptive Neural Networks for Quadruped Motion Control". Then put all the .bvh files into the `data_preprocess/Lafan1_and_dog/DogSet`

**Process the Lafan1 and dog data using the following commands:**

````
python data_preprocess/Lafan1_and_dog/extract.py
```` 
It will use train/test split files in the folder to generate the processed .npz files and the statistic files for training and testing. 

You can also download our preprocessed data from [Google Drive](https://drive.google.com/file/d/1q6xjlssq3G-O-SBr-IHGVJnCCM_KrSCA/view?usp=sharing) and put all the npz files into `data_preprocess/Lafan1_and_dog/` after unzipping.

</details>

### 3. Download pretrained model

<details><summary>Click to expand</summary>

**Model for retargeting between Mixamo characters**

Download the models from [here](https://drive.google.com/file/d/1jYtOLCDye68nShXNlse-I5hCe7ZAHaG-/view?usp=sharing) and unzip the file in the workspace of this project by following command:
````
unzip pretrained_mixamo.zip
````
Eventually the `./pretrained_mixamo` folder should have the following structure:
  ```
   pretrained_mixamo
      └-- models
    	     └-- optimizers
		     └-- topology0
		     └-- topology1
      └-- para.txt   
   ```   

**Model for retargeting between biped and quadruped**

Download the models from [here](https://drive.google.com/file/d/1p-fDC9nIuqktVaqxcAr4wSa09mGq1_63/view?usp=sharing) and unzip the file by following command:

````
unzip pretrained_lafan1dog.zip
````
The `./pretrained_lafan1dog` folder should look like this:
 ```
  pretrained_lafan1dog
    └-- models
            └-- dog
            └-- human
            └-- optimizers
    └-- para.txt   
```   

</details>

## Quick Start
<details><summary>Click to expand</summary>

We provide scripts together with demo examples using files specified in bvh format.

To generate the example of retargeting from biped to quadruped skeleton, run the following command:
````
python demo_hum2dog.py 
````
The retargeting source file and the results will be saved in floder `./pretrained_lafan1dog/demo/hum2dog`

As for retargeting from quadruped to biped, run:
````
python demo_dog2hum.py
````

To generate the retargeting results between Mixamo skeletons, please run:
````
python demo_mixamo.py
````
The results are stored in floder `./pretrained_mixamo/demo` including intra- and cross-strutural retargeting. 

</details>

## Training models from scratch
<details><summary>Click to expand</summary>

**Train models using Mixamo dataset**
````
python train_mixamo.py --save_dir ./pretrained_mixamo --batch_size 128 --model pan --learning_rate 1e-3 --cuda_device cuda --use_parallel True
````

**Train models using Lafan1 and dog datasets**
````
python train_lafan1dog.py --save_dir ./pretrained_lafan1dog --rec_loss_type norm_rec --lambda_cycle 1e-3 --lambda_retar_vel 1e3 --device cuda:0 --batch_size 128 --with_end True
````

</details>

## Quantitative Evaluations

````
python test_mixamo.py --save_dir ./pretrained_mixamo --model pan --epoch 1000
````
This will evaluate the model performance on Mixamo dataset by intra- and cross-structural retargeting. The generated retargeting results will be saved in `./pretrained_mixamo/results`.

## Comments
* Our code for the training architecture and strategy builds on [SAN](https://github.com/DeepMotionEditing/deep-motion-editing).
* The data processing code for Lafan1 and dog is based on the project of [Ubisoft La Forge Animation Dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)

## Citation
If you find this project useful for your research, please consider citing:
````
@article{hu2023pose,
  title={Pose-Aware Attention Network for Flexible Motion Retargeting by Body Part},
  author={Hu, Lei and Zhang, Zihao and Zhong, Chongyang and Jiang, Boyuan and Xia, Shihong},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2023},
  publisher={IEEE}
}
````
