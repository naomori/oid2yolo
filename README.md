# Open Image Dataset から YOLO フォーマットのデータセットを作成する

[Open Images Dataset V6][] から [YOLO format][] 
のデータセットを作成する Python スクリプトを作成します。

## 入力情報

スクリプトに渡す入力ファイルは次に示す２つです。

### 1. Open Images Dataset V6 関連ファイルファイル(`--oid`)

以下の情報が書かれた json ファイルを渡します。

* 画像イメージ tarballs
    - train, val, test
* アノテーションファイル
    - train, val, test
* クラスメタデータファイル

以下が書かれた`yaml`ファイルを入力とします。

### 2. YOLO データセット情報ファイル(`--yolo`)

以下の情報が書かれた yaml ファイルを渡します。

* 画像イメージファイルの置き場所
    - train, val, test
* クラス名リスト
* クラス数
    - 記載したクラス数分だけクラスリストから取り出し、
        アノテーション情報を抽出する対象とします。

#### NOTE1: アノテーションファイルの置き場所について

画像イメージファイルのパスには、必ず`images`ディレクトリが含まれていることを条件とします。
そして、アノテーションファイルのパスは、その`images`を`labels`にしたものとします。

画像ファイルとアノテーションファイルのパスは以下のようになります。

```
/workspace/yolov5/images/train/0a4d092f94a79ef7.jpg # image
/workspace/yolov5/labels/train/0a4d092f94a79ef7.txt # label
```

したがって、画像ファイルのパスに文字列`images`は1つだけ含まれている必要があります。

#### NOTE2: アノテーション情報について

アノーテション情報は、`darknet format`で用意する必要があります。  
`darknet format`では、1つの画像ファイルにつき、1つの`*.txt`ファイルを用意します。
* 1行につき、1つのオブジェクト
* 各行は、以下で構成されます。
    - `class`: names配列のindex
    - `x_center`
    - `y_center`
    - `width`
    - `height`
* それぞれの column は space で区切られます。
* Bouding Box は 0-1 で正規化した値を使います。
* クラス番号は0から開始します(`*.yaml`ファイルで`names`として書いた順番です)。

## 環境構築

```bash
~ ❯❯❯ docker pull ultralytics/yolov5
```

- - -
[Open Images Dataset V6]: https://storage.googleapis.com/openimages/web/index.html
[YOLO format]: https://github.com/AlexeyAB/Yolo_mark/issues/60#issuecomment-401854885
