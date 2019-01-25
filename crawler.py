# -*- coding: utf-8 -*-
# お遊びで使うデータ集め用
# お遊びなのでデータがかぶることもしばしば つまりやばい
# 外れ値も何も考慮してない

import os
from icrawler.builtin import GoogleImageCrawler
import shutil # ファイル移動用
import glob
import argparse

parser = argparse.ArgumentParser(description='dataset crawler')
parser.add_argument('--num_imgs', '-n', type=int, default=50, help='number of dataset images')
option = parser.parse_args()
print(option)

dataset_dir = './data/train/'
dirs = os.listdir(dataset_dir)
num_imgs = option.num_imgs # default = 50 (train = 45, test = 5)
num_test_imgs = int(num_imgs * 0.1)
# print("{0:06d}".format(num_test_imgs)) # 桁合わせ確認

for class_name in dirs:
    if class_name != '.DS_Store':
        key = class_name.replace('_', ' ') # アンダーバーをスペースに置き換え
        print(key)
        train_dir = './data/train/' + class_name + '/'
        crawler = GoogleImageCrawler(storage={"root_dir": train_dir})
        crawler.crawl(keyword=key, max_num=num_imgs)

        # .JPG => .jpg | dataloaderが読み込まないから
        for path in glob.glob(train_dir + '/*.JPG'):
            root, ext = os.path.splitext(path) # ファイル名, 拡張子
            os.rename(path, root + '.jpg')

        # 集めた画像の1割をtestに移動
        test_dir = './data/test/' + class_name + '/'
        os.makedirs(test_dir, exist_ok=True)

        for i in range(1, num_test_imgs+1):
            i = num_imgs - num_test_imgs + i
            search_name = train_dir + '{0:06d}'.format(i) + '.*' # 000xxx.***
            img_path = glob.glob(search_name)[0]
            img_name = os.path.basename(img_path)

            shutil.move(img_path, test_dir + img_name)
