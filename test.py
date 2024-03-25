import argparse, os
import torch
from torch.autograd import Variable
import imageio
from PIL import Image
import numpy as np
import time, math
import matplotlib.pyplot as plt
from vdsr import Net
import torchvision.transforms as transforms
from torch.serialization import SourceChangeWarning
import warnings 
import glob

def main():
    warnings.filterwarnings("ignore", category=SourceChangeWarning)

    parser = argparse.ArgumentParser(description="PyTorch VDSR Demo")
    parser.add_argument("--cuda", action="store_true", help="use cuda?")
    parser.add_argument("--model", default="model/model_epoch_50.pth", type=str, help="model path")
    parser.add_argument("--input_dir", default="datas/", type=str, help="Directory of low-resolution images")
    parser.add_argument("--output_dir", default="result/", type=str, help="Directory to save high-resolution images")
    parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
    parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
    
    opt = parser.parse_args()
    cuda = opt.cuda

    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    # 모델 로드
    model = torch.load(opt.model, map_location='cpu')["model"]

    if opt.cuda:
        model = model.cuda()

    # 디렉토리 존재 유무 확인
    os.makedirs(opt.output_dir, exist_ok=True)

    image_files = glob.glob(os.path.join(opt.input_dir, '*'))
    for image_file in image_files:
        print(f"Processing {image_file}")
        img = Image.open(image_file)

        if img.mode == 'L':
            img_to_tensor = transforms.ToTensor()
            input = img_to_tensor(img).unsqueeze(0)
        else:
            img = img.convert('RGB')
            y, cb, cr = img.convert('YCbCr').split()
            img_to_tensor = transforms.ToTensor()
            input = img_to_tensor(y).unsqueeze(0)

        if opt.cuda:
            input = input.cuda()

        with torch.no_grad():
            start_time = time.time()
            out = model(input)
            elapsed_time = time.time() - start_time
        print(f"Processing time: {elapsed_time:.3f}s")

        out_img_y = out.cpu().squeeze(0).numpy()
        out_img_y = np.clip(out_img_y, 0.0, 1.0) * 255.0
        out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

        if img.mode == 'L':
            out_img = out_img_y
        else:
            out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
            out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
            out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

        output_path = os.path.join(opt.output_dir, os.path.basename(image_file))
        out_img.save(output_path)
        print(f"Saved high-resolution image to {output_path}")

if __name__ == "__main__":
    main()
