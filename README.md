# BoRe-Depth
## A novel lightweight monocular depth estimation method.

It is suitable for all kinds of real-time tasks, especially on embedded devices of unmanned systems.

## Performance
We provide two models (NYUv2 and KITTI) for robust relative depth estimation (We will publish the weights and datas after the paper is received.)
|  **Dataset**  |  **Param./M**  |  **Abs_Rel**  |  **RMSE**  |  **$\delta$<sub>1</sub>**  |  **$\delta$<sub>2</sub>**  |  **$\delta$<sub>3</sub>**  |  **$\epsilon$<sub>DBE</sub><sup>acc</sup>**  |
| :-------: | :-------------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | 
|   NYUv2   |  8.7  |    |
|   KITTI   |  8.7  |    |

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
* ```--dataset_name```: [nyu, kiit, iBims]. The size of the predicted depth map based on the selected dataset.
* ```--dataset_dir```: The path to the test dataset (both ```jpg``` and ```png``` format).
* ```--ckpt_path```: The path to the trained weights.

### Inference
```
python infer.py --dataset_name nyu --ckpt_path checkpoints/nyu.ckpt --input_dir demo --output_dir output --save-vis --save-depth
```
Arguments  
* ```--dataset_name```: [nyu, kiit, iBims]. The size of the predicted depth map based on the selected dataset.
* * ```--ckpt_path```: The path to the trained weights.
* ```--input_dir```: The path to the input picture or folder (both ```jpg``` and ```png``` format).
* ```--output_dir```: The path to the output depth map (both ```png``` and ```npy``` format).
* ```--save-vis```: Saving the visual images.
* ```--save-depth```: Saving the numpy results.
