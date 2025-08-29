# [IROS2025]BoRe-Depth
## A novel lightweight monocular depth estimation method.

It is suitable for all kinds of real-time tasks, especially on embedded devices of unmanned systems.

## Performance

We provide [two models (NYUv2 and KITTI)](https://drive.google.com/drive/folders/11XxOXqWKp3bXe2Sv2ED90__S3GwkgMRI) for robust relative depth estimation. 
|  **Dataset**  |  **Param./M**  |  **Abs_Rel**  |  **RMSE**  |  **$\delta$<sub>1</sub>**  |  **$\delta$<sub>2</sub>**  |  **$\delta$<sub>3</sub>**  |  **$\epsilon$<sub>DBE</sub><sup>acc</sup>**  |
| :-------: | :-------------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | 
|   NYUv2   |  8.7  |  0.101  |  0.429  |  0.883  |  0.971  |  0.993  |  2.083  |
|   KITTI   |  8.7  |  0.103  |  4.323  |  0.889  |  0.967  |  0.986  |  2.649  |

**Indoor Scene Visualization**
![Image](https://github.com/user-attachments/assets/a655a29c-167d-4935-80c5-262a3e928b8e)

**Outdoor Scene Visualization**
![Image](https://github.com/user-attachments/assets/27c6e306-ad8e-49b1-b058-ee70dea1c5f5)


## Usage
### Installation
```
git clone https://github.com/liangxiansheng093/BoRe-Depth.git
cd BoRe-Depth
pip install -r requirements.txt
```
Download weights to ```checkpoints``` folder.

### Test
```
python test.py --dataset_name nyu --dataset_dir datasets/nyu/testing --ckpt_path checkpoints/nyu.ckpt
```
Options:
* ```--dataset_name```: [nyu, kiit, iBims]. The size of the predicted depth map based on the selected dataset.
* ```--dataset_dir```: The path to the test dataset (both ```jpg``` and ```png``` format).
* ```--ckpt_path```: The path to the trained weights.

### Inference
```
python infer.py --dataset_name nyu --ckpt_path checkpoints/nyu.ckpt --input_dir demo --output_dir output --save-vis --save-depth
```
Options:
* ```--dataset_name```: [nyu, kiit, iBims]. The size of the predicted depth map based on the selected dataset.
* ```--ckpt_path```: The path to the trained weights.
* ```--input_dir```: The path to the input picture or folder (both ```jpg``` and ```png``` format).
* ```--output_dir```: The path to the output depth map (both ```png``` and ```npy``` format).
* ```--save-vis```: Saving the visual images.
* ```--save-depth```: Saving the numpy results.
