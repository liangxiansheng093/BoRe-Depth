from tqdm import tqdm
import torch
from imageio.v2 import imread, imwrite
from path import Path
import os

from configs.configs import get_opts
from model.BoRe_Depth import BoRe_Depth
from datasets.dataset import inference_transform
from visualization import *


@torch.no_grad()
def main():
    hparams = get_opts()
    system = BoRe_Depth()
    system = system.load_from_checkpoint(hparams.ckpt_path, strict=False)

    # model set
    model = system.depth_net
    model.cuda()
    model.eval()

    # input and output path
    input_dir = Path(hparams.input_dir)
    output_dir = Path(hparams.output_dir)
    output_dir.makedirs_p()

    if hparams.save_vis or hparams.save_depth:
        output_dir.makedirs_p()

    image_files = sum([(input_dir).files('*.{}'.format(ext))
                      for ext in ['jpg', 'png']], [])
    image_files = sorted(image_files)

    print('{} images for inference'.format(len(image_files)))

    for i, img_file in enumerate(tqdm(image_files)):

        filename = os.path.splitext(os.path.basename(img_file))[0]

        img = imread(img_file).astype(np.float32)
        tensor_img = inference_transform([img])[0][0].unsqueeze(0).cuda()
        pred_depth = model(tensor_img)

        if hparams.save_vis:
            vis = visualize_depth(pred_depth[0, 0]).permute(
                1, 2, 0).numpy() * 255
            imwrite(output_dir/'{}.jpg'.format(filename),
                    vis.astype(np.uint8))

        if hparams.save_depth:
            depth = pred_depth[0, 0].cpu().numpy()
            np.save(output_dir/'{}.npy'.format(filename), depth)


if __name__ == '__main__':
    main()
