import torch
import torch.nn as nn
from math import sqrt

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        # 3x3 컨볼루션 레이어
        # 입력 채널 수: 64, 출력 채널 수: 64,  커널 크기: 3
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # relu 활성화 함수 정의
        # inplace = true: 연산을 원본 텐서에 직접 적용 
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # 입력을 컨볼루션 레이어와 ReLU 활성화 함수에 통과시킨다. 
        return self.relu(self.conv(x))
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 잔차 블록 정의
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        # 입력 이미지의 채널을 64로 확장하는 3x3 컨볼루션 레이어를 정의
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # 출력 이미지의 채널을 1로 축소하는 컨볼루션 레이어를 정의
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # 네트워크의 모든 컨볼루션 레이어의 가중치 초기화를 수행
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    # 잔차 블록 생성             
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers) # 리스트에 있는 모든 블록을 연결하여 하나의 시퀀셜 레이어 생성 

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out) # 최종 출력 
        out = torch.add(out,residual)
        return out
    # 입력 이미지 x를 받고, 이를 residual 변수에 복사하여 저장. 그 다음 입력 이미지를 첫 번째 컨볼루션 레이어와 ReLU 활성화 함수에 통과시킨다. 