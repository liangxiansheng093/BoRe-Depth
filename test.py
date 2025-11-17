import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from path import Path
from imageio.v2 import imread
import torch.nn.functional as F

from datasets.dataset import inference_transform
from configs.configs import get_opts, get_size
from model.BoRe_Depth import BoRe_Depth
from visualization import *


def crawl_folder(folder, dataset='nyu'):

    imgs = sorted((folder/'color/').files('*.png') + (folder/'color/').files('*.jpg'))
    depths = sorted((folder/'depth/').files('*.png'))

    return imgs, depths


class TestSet(Dataset):
    def __init__(self, root, transform=None, dataset='nyu'):
        self.root = Path(root)/'testing'
        self.transform = transform
        self.dataset = dataset
        self.imgs, self.depths = crawl_folder(self.root, self.dataset)

    def __getitem__(self, index):
        img = imread(self.imgs[index]).astype(np.float32)
        depth = torch.from_numpy(imread(self.depths[index]).astype(np.float32)).float()/1000

        if self.transform is not None:
            img, _ = self.transform([img], None)
            img = img[0]

        return img, depth

    def __len__(self):
        return len(self.imgs)


@torch.no_grad()
def compute_errors(gt, pred, dataset):
    abs_diff = abs_rel = sq_rel = log10 = rmse = rmse_log = a1 = a2 = a3 = 0.0

    batch_size, h, w = gt.size()

    if pred.nelement() != gt.nelement():
        pred = F.interpolate(
            pred, [h, w], mode='bilinear', align_corners=False)

    pred = pred.view(batch_size, h, w)

    if dataset == 'kitti':
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1, x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1
        max_depth = 80

    if dataset == 'nyu':
        crop_mask = gt[0] != gt[0]
        crop = np.array([45, 471, 41, 601]).astype(np.int32)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        max_depth = 10

    min_depth = 0.1
    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > min_depth) & (current_gt < max_depth)
        valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid]

        # align scale
        valid_pred = valid_pred * \
            torch.median(valid_gt)/torch.median(valid_pred)

        valid_pred = valid_pred.clamp(min_depth, max_depth)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        diff_i = valid_gt - valid_pred
        abs_diff += torch.mean(torch.abs(diff_i))
        abs_rel += torch.mean(torch.abs(diff_i) / valid_gt)
        sq_rel += torch.mean(((diff_i)**2) / valid_gt)
        rmse += torch.sqrt(torch.mean(diff_i ** 2))
        rmse_log += torch.sqrt(torch.mean((torch.log(valid_gt) -
                               torch.log(valid_pred)) ** 2))
        log10 += torch.mean(torch.abs((torch.log10(valid_gt) -
                            torch.log10(valid_pred))))

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, log10, rmse, rmse_log, a1, a2, a3]]


@torch.no_grad()
def main():
    hparams = get_opts()

    # initialize network
    system = BoRe_Depth()

    # load ckpts
    system = system.load_from_checkpoint(hparams.ckpt_path, strict=False)

    model = system.depth_net
    model.cuda()
    model.eval()

    test_dataset = TestSet(
        hparams.input_dir,
        transform=inference_transform,
        dataset=hparams.dataset_name
    )
    print('{} samples found in test scenes'.format(len(test_dataset)))

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=8,
                             pin_memory=True
                             )

    all_errs = []
    for i, (tgt_img, gt_depth) in enumerate(tqdm(test_loader)):
        pred_depth = model(tgt_img.cuda())

        errs = compute_errors(gt_depth.cuda(), pred_depth,
                              hparams.dataset_name)

        all_errs.append(np.array(errs))

    all_errs = np.stack(all_errs)
    mean_errs = np.mean(all_errs, axis=0)

    print("\n  " + ("{:>8} | " * 9).format("abs_diff", "abs_rel",
          "sq_rel", "log10", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 9).format(*mean_errs.tolist()) + "\\\\")


if __name__ == '__main__':
    main()
