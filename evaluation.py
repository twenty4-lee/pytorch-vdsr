import argparse
import numpy as np
import math
from PIL import Image
import torchvision.transforms as transforms

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def main():
    parser = argparse.ArgumentParser(description="Image PSNR Evaluation")
    parser.add_argument("--image1", default="data/contrast_1-1.png", type=str, help="Path to the first image")
    parser.add_argument("--image2", default="result/contrast_1-1.png", type=str, help="Path to the second image")
    opt = parser.parse_args()

    # 이미지 로드 및 Y 채널 추출
    img1 = Image.open(opt.image1).convert('YCbCr').split()[0]
    img2 = Image.open(opt.image2).convert('YCbCr').split()[0]

    # 이미지를 텐서로 변환
    to_tensor = transforms.ToTensor()
    img1_tensor = to_tensor(img1)
    img2_tensor = to_tensor(img2)

    # 두 이미지 간의 PSNR 계산
    psnr_value = PSNR(img1_tensor.numpy()*255.0, img2_tensor.numpy()*255.0)
    print(f"PSNR between the two images: {psnr_value} dB")

if __name__ == "__main__":
    main()