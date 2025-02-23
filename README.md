# BoRe-Depth
## A novel lightweight monocular depth estimation method.

It is suitable for all kinds of real-time tasks, especially on embedded devices of unmanned systems.

## Performance
We provide two models (NYUv2 and KITTI) for robust relative depth estimation:
| :**Dataset**: | :**Param./M**: | :**Abs_Rel**: | :**RMSE**: | :**$\delta$<sub>1</sub>**: | :**$\delta$<sub>2</sub>**: | :**$\delta$<sub>3</sub>**: | :**$\epsilon$<sub>DBE</sub><sup>acc</sup>**: |
| ------- | ------------- | ------- | ------- | ------- | ------- | ------- | ------- | 
|  :NYUv2:  | :8.7: | :0.101: | :0.429: | :0.883: | :0.971: | :0.993: | :2.083: |
|  :KITTI:  | :8.7: | :0.103: | :4.323: | :0.889: | :0.967: | :0.986: | :2.649: |

## Usage
### Installation
```
git clone https://github.com/liangxiansheng093/BoRe-Depth.git
cd BoRe-Depth
pip install -r requirements.txt
```
Download [weights](https://drive.google.com/drive/my-drive?dmr=1&ec=wgc-drive-globalnav-goto) to ```checkpoints``` folder.


### Inference
```
python infer.py --dataset_name nyu --input_dir demo --output_dir output --save-vis --save-depth
```
Arguments: 
* ```--dataset_name```: [nyu, kiit, iBims]. The size of the predicted depth map based on the selected dataset.
* ```--input_dir```: The path to the input picture or folder (both ```jpg``` and ```png``` format images).
