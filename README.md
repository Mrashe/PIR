# PIR
few shot image generation via style adaptation and content preservation
#Overview
<img src='idea.png'/>

Our method help align the spatial structural information between source and target GAN to assist adaption.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)

## Requirements

**Note:** The base model is taken from [Few-shot-gan-adaptation](https://github.com/WisconsinAIVision/few-shot-gan-adaptation)'s implementation from [@WisconsinAIVision](https://github.com/WisconsinAIVision)

- Linux
- PyTorch 1.12.0
- Python 3.8
- Install all the libraries through `pip install -r requirements.txt`

### Sample images from a model

We provide the pre-trained models for different source and target GAN models. Download the model from [Here](https://drive.google.com/drive/folders/1v3Ge9uGqY294vFqcwqQIgznxgtrej6bm?usp=sharing).

To generate images from a pre-trained GAN, run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python generate.py --ckpt_target /path/to/model/
```

This will save the images in the `test_samples/` directory.

## Training (adapting) your own GAN

- Raw data should be organized as:
```
├── raw_data
│   | <dataset_name> 
│     ├── images
│       ├── 000.png
        ├── 001.png
        ├── ...
│    
```
- Run `python prepare_data.py --out processed_data/<dataset_name> --size 256 ./raw_data/<dataset_name>`. This will generate the processed version of the data in `./processed_data` directory. 

- If you wish to use some other source model, make sure that it follows the generator architecture defined in this [pytorch implementation](https://github.com/rosinality/stylegan2-pytorch) of StyleGAN2, or you can modify the generator's architecture in `models.py` accordingly.

- Run the following command to adapt the source GAN (e.g. FFHQ) to the target domain (e.g. sketches):

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --ckpt /path/to/source_model --data_path /path/to/target_data --exp <exp_name>

# sample run
CUDA_VISIBLE_DEVICES=0 python train.py --ckpt ./checkpoints/source_ffhq.pt --data_path ./processed_data/sketches --exp ffhq_to_sketches    
```
This will create directories with name `ffhq_to_sketches` in `./checkpoints/` (saving the intermediate models) and in `./samples` (saving the intermediate generated images). 

