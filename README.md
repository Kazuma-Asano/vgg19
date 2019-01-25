# VGG19 初学者用プログラム in Pytorch
mnistに飽きた，自分で作ったデータセットで学習，識別したい，VGGよくわからんなどの人向け  
Pytorchに実装されているVGGシリーズのプログラムは洗礼されていて，正直初学者である私には分かりづらい事があった．   
なので，全部小分けして実装！（だけどその分プログラムが長いので，あくまでイメージを掴む用）  

ついでに自動的に画像を集めてくれるプログラムも実装しておきました．（おもちゃ程度に考えてください）  
デフォルトでは，86やGT-Rなどの国産スポーツカーを識別するフォルダ構成になっています．  

* 環境
  * macOSで実行確認
  * TODO:Ubuntuもしておく

* ライプラリ (実装時)
  * `Python = 3.5`
  * `Pytorch = 1.0.0`
  * `icrawler = 0.6.2` （自動画像集め用）
  その他いろいろ

* 使い方
  1. `/data/train/`に識別したい画像クラス名のフォルダを作成
  2. `$python crawler.py`を実行して画像集め
  3. `$python train.py`を実行して学習(`$python train.py --cuda`でGPUを利用して学習)
  4. `$python test.py`で確認 （一応，学習時にも実行されます）

* 学習したモデルに対し，新たに集めた画像を入力したい場合
  そのうち実装しておきます
