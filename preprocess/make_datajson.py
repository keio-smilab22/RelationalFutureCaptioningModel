import json
import os
from pathlib import Path

from classopt import classopt, config
from numpy import NaN
from tqdm import tqdm
import pandas as pd

@classopt(default_long=True)
class Args:
    bilas_path: str = config(long="--bilas", short='-b',  default="data/BilaS/")
    ponnet_path: str = config(long='--ponnet', short='-p', default='data/Ponnet/')
    output_path: str = config(long='--output', short='-o',  default="data/BilaS/bilas.jsonl")
    numSet: int = config(long='--numSet', short='-n', default=4, choices=[1,3,4])
    att_type: str = config(long='--att_type', short='-t', default='att', choices=['gray', 'att', 'over', 'ave'])

def main():
    """
    Billa-Sのdata関連をまとめたjsonファイルを作成する
    
    Contents:
        image_rgb: カメラ視点のRGB画像へのpath
        image_depth: カメラ視点のdepth画像へのpath
        target_rgb: 対象物体のrgb画像へのpath
        target_depth: 対象物体のdepth画像へのpath
        attention_map: Ponnetの出力attention map画像へのpath
        sentence: 説明文
    """

    args: Args = Args.from_args()

    billas = dict()

    # sentenceを基準に他のデータをとってくる
    sentence_path = Path(args.bilas_path, f'S-set{args.numSet}', 'sentences/', f'sentences_S-set{args.numSet}.csv')
    sentence_df = pd.read_csv(sentence_path)
    H, _ = sentence_df.shape

    path_df = pd.read_csv(Path(args.ponnet_path, f'S-set{args.numSet}', "path.csv"), header=None)
    columns = ['image_rgb', 'image_depth', 'target_rgb', 'target_depth', 'bbox', 'feature']
    path_df.columns = columns

    att_df = pd.read_csv(Path(args.ponnet_path, f'S-set{args.numSet}', "att_path.csv"), header=None)
    columns = ['scene', 'gray_rgb', 'gray_depth', 'att_rgb', 'att_depth', 'over_rgb', 'over_depth', 'att_ave']
    att_df.columns = columns

    for i in tqdm(range(H)):
        scene, sentence = sentence_df['Scene'][i], sentence_df['Sentence'][i]
        if sentence is NaN:
            continue

        # カメラ視点の像像
        image_rgb = path_df.iloc[scene-1, :]['image_rgb']
        image_depth = path_df.iloc[scene-1, :]['image_depth']

        # 対象物体の画像
        target_rgb = path_df.iloc[scene-1, :]['target_rgb']
        target_depth = path_df.iloc[scene-1, :]['target_depth']

        # attention map
        if args.att_type == 'gray':
            att_rgb = att_df.iloc[scene-1, :]['att_gray_rgb']
            att_depth = att_df.iloc[scene-1, :]['att_gray_depth']
        if args.att_type == 'att':
            att_rgb = att_df.iloc[scene-1, :]['att_rgb']
            att_depth = att_df.iloc[scene-1, :]['att_depth']
        if args.att_type == 'over':
            att_rgb = att_df.iloc[scene-1, :]['over_rgb']
            att_depth = att_df.iloc[scene-1, :]['over_depth']
        if args.att_type == 'ave':
            att_rgb = att_df.iloc[scene-1, :]['att_ave']
            att_depth = att_df.iloc[scene-1, :]['att_ave']

        billas['scene'] = str(scene)
        billas['image_rgb'] = image_rgb
        billas['image_depth'] = image_depth
        billas['target_rgb'] = target_rgb
        billas['target_depth'] = target_depth
        billas['attention_map_rgb'] = att_rgb
        billas['attention_map_depth'] = att_depth
        billas['sentence'] = sentence
        
        with open(args.output_path, 'a') as f:
            f.write(json.dumps(billas, ensure_ascii=False))
            f.write('\n')

if __name__ == "__main__":
    main()