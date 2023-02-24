import os
import torch
from torchvision import transforms
import pandas as pd
from PIL import Image


# 1. 해당 폴더 안의 데이터 파일 경로를 리스트에 담아두는 과정 필요
def make_file_list():
    train_img_list = list()
    
    for img_idx in range(200):
        img_path = "./Your/data_1/path" + str(img_idx)+'.jpg'
        train_img_list.append(img_path)

        img_path = "./Your/data_2/path" + str(img_idx)+'.jpg'
        train_img_list.append(img_path)

    return train_img_list

# 2. 이미지 전처리
class ImageTransform():
    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    def __call__(self, img):
        return self.data_transform(img)

# 3. Dataset을 상속받아 나만의 데이터셋 인스턴스를 생성해주는 클래스 구현
class Img_Dataset(data.Dataset):
    def __init__(self, file_list, transform):
        self.file_list = file_list,
        self.transform = transform
    
    def __len__(self):
        return len(self,file_list)
    
    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        
        # 라벨이 있는 경우 파일명으로부터 라별명 추출하여 라벨 추가
        
        return img_transformed
    
# 4. 마지막으로, DataLoader에 나만의 데이터셋 넣기
train_img_list = make_file_list()

mean = (0.5,)
std = (0.3, )

train_dataset = Img_Dataset(file_list = train_img_list, 
                            transform = ImageTransform(mean, std))

train_dataloader = data.DataLoader(train_dataset, 
                                   batch_size=64,
                                   shuffle=True)

batch_iterator = iter(train_dataloader)
images = next(batch_iterator)

print(images.size())
    