import os
import csv
import json
import math
import re
from glob import glob
from pathlib import Path
from statistics import mean, stdev

import nltk
import MeCab
from classopt import classopt, config
from tqdm import tqdm
from numpy import NaN, nan
import pandas as pd

"""
前提:
    data/Ponnet/ に使用するS-setNフォルダが含まれており、
    各S-setNフォルダ内に
        ・attentions
        ・csv_bbox
        ・csv_feature
        ・ponNet01 ~ ponNet20
        ・path.csv
    を格納していることを前提とする
    
    また、
    data/Bilas/S-setNフォルダ内の
        ・sentence_S-setN.csv
    data/Bilas/フォルダ内の
        ・bilas_all.jsonl | ・bilas_all_mecab.jsonl
        ・bilas_{train,valid,test}_mecab.jsonl
        ・caption_{valid, test}.json
    がない(または削除されている)ことを前提とする

パーサー用のライブラリ
    # pip install mecab-python3==1.0.3
    # pip install unidic-lite==1.0.8

実行コマンド
    python preprocess/do_all_preprocess.py --useSet 3 4 --delete_two_text --save_csv_set --att_type over --save_all_csv --save_all_json --save_mecab_json --seed 42 --save_split_json --save_caption_json --calc_max_t_len --top_k 10
"""

@classopt(default_long=True)
class Args:
    useSet: int = config(long='--useNum', short="-u", nargs="*", default=[4])
    # path to dataset file
    data_dir: str = config(long="--dir", short='-d', default='data/BilaS/')
    ponnet_path: str = config(long='--ponnet', short='-p', default='data/Ponnet/')
    
    # (1) attention pathの作成で使用
    pon_att_num: int = config(long='--attNum', short="-a", default=2000)
    save_att_path: bool = False
    do_att_path: bool = False

    # (2) extrace_sheet2csvで使用
    verNum: int = config(long='--ver', short='-v', default=1, choices=[1,2])
    delete_list: str = [".", "?", "!", "。", "？", "！"]
    delete_two_text: bool = config(long='--delete', short='-del', default=False)
    csv_output_prefix: str = config(long="--csvPrefix", short="-c", default="sentences_")
    save_csv_set: bool = False
    do_csv_set: bool = True

    # (3) make_jsonlで使用
    att_type: str = config(long='--att_type', short='-t', default='att', choices=['gray', 'att', 'over', 'ave'])
    save_all_json: bool = False
    save_all_csv: bool = False
    name_jsonl: str = config(long="--jname", short='-jn', default="bilas_all.jsonl")
    do_all_json: bool = True

    # (4) exec_mecabで使用
    name_after_mecab: str = config(long="--mjname", short='-mjn', default="bilas_all_mecab.jsonl")
    save_mecab_json: bool = False
    do_mecab_json: bool = True

    # (5) create_split_jsonlで使用
    seed: int = 0
    add_mode: bool = False
    add_mode_file: str = config(long='--addname', short='-adname', default="bilas_all_split.jsonl")
    save_split_json: bool = False
    do_split_json: bool = True
    
    # (6) make_captionで使用
    save_caption_json: bool = False
    do_caption_json: bool = True
    
    # (7) calc_topk_lengthで使用
    calc_max_t_len: bool = False
    top_k: int = config(long='--topk', short='-tk', default=10)
    do_calc_t_len: bool = True

def delete_two_sentence(text):
    """
    'また'を含む文がある場合、
    'また'以降の文を排除して1文にする処理
    """
    text = str(text)
    if re.search(r'。また、', text):
        text = re.sub(r"。また、.*", r"", text)

    return text

def create_pon_att_path(args: Args):
    """
    Summary:
        Ponnetフォルダ内のattenton画像へのpath(csv, jsonl)を作成する
    Args:
        args.useSet: 使用するS-setNを指定 default = 4
        args.ponnet_path : data/Ponnet/
        args.pon_att_num : attentionの画像の枚数 default=2000
        args.save_att_path : att_pathを作成するか否か
    """
    for num in args.useSet:
        input_path = os.path.join(args.ponnet_path, f"S-set{num}")
        output_json = os.path.join(input_path, "att_path.jsonl")
        output_csv = os.path.join(input_path, "att_path.csv")

        if os.path.isfile(Path(input_path, output_json)):
            print("File already exists ...")
            flag = input("Delete existed file ? >> yes:1 no:0 : ")
            if flag == 1:
                os.remove(Path(input_path, output_json))
                os.remove(Path(input_path, output_csv))
            else:
                print("Exit ...")
                return
        
        for id in tqdm(range(1, args.pon_att_num+1), desc="Create Ponnet Attention path ..."):
            attentions = sorted(glob(input_path+"attentions/"+f"*_{id}_*.jpg"))
            assert len(attentions) == 6

            att_gray_rgb = str(Path('/'.join(attentions[1].split('/')[-3:])))
            att_gray_depth = str(Path('/'.join(attentions[0].split('/')[-3:])))

            att_map_rgb = str(Path('/'.join(attentions[3].split('/')[-3:])))
            att_map_depth = str(Path('/'.join(attentions[2].split('/')[-3:])))

            over_rgb = str(Path('/'.join(attentions[5].split('/')[-3:])))
            over_depth = str(Path('/'.join(attentions[4].split('/')[-3:])))
            
            att_ave = str(Path("/".join(glob(input_path+'attentions/'+f'*_{id}.jpg')[0].split('/')[-3:])))

            # for json
            att_dict = {}
            att_dict["id"] = id
            att_dict["att_gray_rgb"] = att_gray_rgb
            att_dict["att_gray_depth"] = att_gray_depth
            att_dict["att_map_rgb"] = att_map_rgb
            att_dict["att_map_depth"] = att_map_depth
            att_dict["over_rgb"] = over_rgb
            att_dict["over_depth"] = over_depth
            att_dict["att_ave"] = att_ave
            
            if args.save_att_path:
                with open(output_json, "a") as wf:
                    wf.write(json.dumps(att_dict, ensure_ascii=False))
            
            # for csv
            if args.save_att_path:
                att_csv = [id, att_gray_rgb, att_gray_depth, att_map_rgb, att_map_depth, over_rgb, over_depth, att_ave]
                with open(output_csv, 'a') as wf:
                    writer = csv.writer(wf)
                    writer.writerow(att_csv)

def extract_sheet2csv(args: Args):
    """
    summary:
        sheetからダウンロードしたcsvから必要なデータだけを抽出したcsvを作成
    
    Args:
        args.useSet          : 使用するS-setN default = 4
        args.data_dir        : BilaSへのpath default='data/BilaS'
        
        args.verNum          : 使用するSentenceVerNを指定 default=1
        args.delete_list     : delete時に削除する余分な文字を指定
        args.delete_two_text : また、以降を削除するか否か
        args.save_csv_set    : sheetから抽出したcsvを作成する
        args.csv_output_prefix: csvファイルの接頭辞 default='sentence_'

    """
    for num in args.useSet:
        sheet_path = os.path.join(args.data_dir, f"S-set{num}", f"S-set{num}_all.csv")

        df = pd.read_csv(sheet_path)
        df = df[2:]

        # 衝突判定ありのみ抽出
        df_col = df[df["isColl"] == 1]
        UseVer = f"SentenceVer{args.verNum}"
        
        if args.delete_two_text:
            df_col[UseVer] = df_col[UseVer].map(delete_two_sentence)

        df_col = df_col[["Scene", UseVer]]
        
        # change column name
        columns = ["Scene", "Sentence"]
        df_col.columns = columns

        if args.save_csv_set:
            output_name = args.csv_output_prefix + f"S-set{num}.csv"
            output_file = os.path.join(args.data_dir, f"S-set{num}", output_name)
            df_col.to_csv(output_file, index=False)

def make_jsonl(args: Args):
    """
    Summary:
        使用するcsvファイルから全データ情報を含めたjsonlファイルを作成する
    Args:
        args.useSet         : 使用するデータセットの指定
        args.data_dir       : Bilasへのpath : default='data/BilaS'
        args.ponnet_path    : Ponnetへのpath : default='data/Ponnet'
        
        args.att_type       : 使用するattention typeの指定
        args.save_all_jsonl : 全データ情報を集めたjsonlを作成する
        args.save_all_csv   : 全データ情報を集めたcsvを作成する(未実装)
        args.name_jsonl     : jsonlファイルの名前
    """
    bilas = dict()
    
    for num in args.useSet:
        # sentenceを基準に必要なponnetの出力画像を取ってくる
        sentence_path = os.path.join(args.data_dir, f"S-set{num}", f"sentences_S-set{num}.csv")
        df_text = pd.read_csv(sentence_path)
        H, _ = df_text.shape

        path_df = pd.read_csv(Path(args.ponnet_path, f"S-set{num}", "path.csv"), header=None)
        columns = ['image_rgb', 'image_depth', 'target_rgb', 'target_depth', 'bbox', 'feature']
        path_df.columns = columns
        
        df_att = pd.read_csv(Path(args.ponnet_path, f"S-set{num}", "att_path.csv"), header=None)
        columns = ['scene', 'gray_rgb', 'gray_depth', 'att_rgb', 'att_depth', 'over_rgb', 'over_depth', 'att_ave']
        df_att.columns = columns
        
        for i in tqdm(range(H)):
            scene, sentence = df_text['Scene'][i], df_text['Sentence'][i]
            if sentence is NaN or sentence is nan:
                continue

            # カメラ視点の像像
            image_rgb = path_df.iloc[scene-1, :]['image_rgb']
            image_depth = path_df.iloc[scene-1, :]['image_depth']

            # 対象物体の画像
            target_rgb = path_df.iloc[scene-1, :]['target_rgb']
            target_depth = path_df.iloc[scene-1, :]['target_depth']

            # attention map
            if args.att_type == 'gray':
                att_rgb = df_att.iloc[scene-1, :]['att_gray_rgb']
                att_depth = df_att.iloc[scene-1, :]['att_gray_depth']
            if args.att_type == 'att':
                att_rgb = df_att.iloc[scene-1, :]['att_rgb']
                att_depth = df_att.iloc[scene-1, :]['att_depth']
            if args.att_type == 'over':
                att_rgb = df_att.iloc[scene-1, :]['over_rgb']
                att_depth = df_att.iloc[scene-1, :]['over_depth']
            if args.att_type == 'ave':
                att_rgb = df_att.iloc[scene-1, :]['att_ave']
                att_depth = df_att.iloc[scene-1, :]['att_ave']
            
            # S-setN
            setNum = image_rgb.split("/")[0]

            bilas['setNum'] = str(setNum[-1])
            bilas['scene'] = str(scene)
            bilas['image_rgb'] = image_rgb
            bilas['image_depth'] = image_depth
            bilas['target_rgb'] = target_rgb
            bilas['target_depth'] = target_depth
            bilas['attention_map_rgb'] = att_rgb
            bilas['attention_map_depth'] = att_depth
            bilas['sentence'] = sentence

            if args.save_all_json:
                with open(args.data_dir+args.name_jsonl, 'a') as f:
                    f.write(json.dumps(bilas, ensure_ascii=False))
                    f.write('\n')


def exex_mecab(args: Args):
    """
    Summary:
        作成したjsonlファイルにmecabを適用したparse_sentenceを追加したjsonlを作成
    Args:
        args.data_dir         : BilaSへのpath default='data/BilaS'
        args.name_jsonl       : jsonlのファイル名
        
        args.name_after_mecab : mecab適用後のjsonlのファイル名
        args.save_mecab_json  : mecabの適用後ファイルを作成するかどうか
    """
    m = MeCab.Tagger('-Owakati')
    json_file_path = os.path.join(args.data_dir, args.name_jsonl)
    output_file = os.path.join(args.data_dir, args.name_after_mecab)

    with open(json_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            sentence = line['sentence']

            m_sentence = m.parse(sentence)
            m_sentence = m_sentence.replace(' \n', '')
            
            line['parse_sentence'] = m_sentence
            if args.save_mecab_json:
                with open(output_file, "a") as f:
                    json.dump(line, f, ensure_ascii=False)
                    f.write('\n')


def create_split_jsonl(args: Args):
    """
    Summary:
        全データ情報を含むjsonlからtrain/valid/testに分割したjsonlファイルを作成
    Args:
        args.data_dir         : BilaSへのpath default='data/BilaS'
        args.name_after_mecab : mecab適用後のファイルへのpath

        args.seed             : 分割の際のシャッフル時のseedを指定 default=0
        args.add_mode         : mode情報を加えたjsonlを作成する(未実装)
        args.add_mode_file    : mode情報を加えたjsonlのファイル名
        args.save_split_json  : 分割ファイルを作成するか否かを指定
    Input:
        'data/BilaS/bilaS/bilas_all_mecab.jsonl'
    Output:

    """
    json_file_path = os.path.join(args.data_dir, args.name_after_mecab)
    df = pd.read_json(json_file_path, orient="records", lines=True)
    H, _ = df.shape
    # 行をシャッフルする
    df = df.sample(frac=1, ignore_index=True, random_state=args.seed)
    trainNum = math.ceil(H * 0.8)
    valNum = math.ceil(H * 0.1)
    testNum = H - trainNum - valNum

    df_train = df.iloc[:trainNum]
    assert trainNum == df_train.shape[0]
    df_val = df.iloc[trainNum:trainNum+valNum]
    assert valNum == df_val.shape[0]
    df_test = df.iloc[trainNum+valNum:]
    assert testNum == df_test.shape[0]
    
    if args.add_mode:
        modes = [['train']*trainNum + ["valid"]*valNum + ["test"]*testNum]
        modes = modes[0]
        df["mode"] = modes
        output_path = Path(args.data_dir, args.add_mode_file)
    
    if args.save_split_json:
        # train
        output_json_file = Path(args.data_dir + "bilas_train_mecab.jsonl")
        bilas = dict()
        columns = df_train.columns
        for idx in range(df_train.shape[0]):
            for i in range(len(columns)):
                bilas[columns[i]] = str(df_train.iloc[idx,:][columns[i]])

            with open(output_json_file, 'a') as f:
                f.write(json.dumps(bilas, ensure_ascii=False))
                f.write('\n')
        
        # valid
        output_json_file = Path(args.data_dir, "bilas_valid_mecab.jsonl")
        bilas = dict()
        columns = df_val.columns
        for idx in range(df_val.shape[0]):
            for i in range(len(columns)):
                bilas[columns[i]] = str(df_val.iloc[idx,:][columns[i]])

            with open(output_json_file, 'a') as f:
                f.write(json.dumps(bilas, ensure_ascii=False))
                f.write('\n')
        
        # test
        output_json_file = Path(args.data_dir, "bilas_test_mecab.jsonl")
        bilas = dict()
        columns = df_test.columns
        for idx in range(df_test.shape[0]):
            for i in range(len(columns)):
                bilas[columns[i]] = str(df_test.iloc[idx,:][columns[i]])

            with open(output_json_file, 'a') as f:
                f.write(json.dumps(bilas, ensure_ascii=False))
                f.write('\n')


def make_caption(args: Args):
    """
    Summary:
        validとtestの評価時に使用するcaptionのjsonファイルを作成
    Args:
        args.data_dir          : BilaSへのpath('data/BilaS')
        args.save_caption_json : captionファイルを作成するか否か
    Input:
        data/BilaS/bilas_{valid, test}_mecab.jsonl
    Ouput:
        data/BilaS/caption_{valid, test}.json
    """
    modes = ["train", "valid", "test"]
    for mode in modes:
        input_path = Path(args.data_dir, f"bilas_{mode}_mecab.jsonl")
        output_path = Path(args.data_dir, f"caption_{mode}.json")
        with open(input_path) as f:
            lines = f.read()
            lines = lines.split('\n')
        
        captions = dict()
        for line in lines:
            if len(line) == 0:
                continue
            data = json.loads(line)
            ids = data["setNum"] + "_" + data['scene']
            captions[ids] = data['parse_sentence']
            

        if args.save_caption_json:
            with open(output_path, 'a') as f:
                f.write(json.dumps(captions, ensure_ascii=False))


def calc_topk_length(args: Args):
    """
    Summary:
        trainに含まれるデータからtop-k(less-k)のlengthを経産する
    Args:
        args.data_dir : BilaSへのpath(data/BilaS)
    Input:
        data/BilaS/bilas_train_mecab.jsonl
    """
    if args.calc_max_t_len:
        input_file = Path(args.data_dir, "bilas_train_mecab.jsonl")

        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        lengths = []
        for line in tqdm(lines, desc="Calc Top-k length..."):
            line = json.loads(line)
            text = nltk.tokenize.word_tokenize(line['parse_sentence'])
            lengths.append(len(text))
        
        lengths = sorted(lengths, reverse=True)

        print(f"Result Top{args.top_k} Length >> {lengths[:args.top_k]}")
        print(f"Show Less Top{args.top_k} Length >> {lengths[-args.top_k:]}")
        print(f"Show Mean Length >> {mean(lengths)}")
        print(f"Show Std  Length  >> {stdev(lengths)}")


def main():
    args: Args = Args.from_args()

    # (1) Ponnetフォルダ内のattention画像へのpath(csv)を作成する
    if args.do_att_path:
        create_pon_att_path(args)

    # (2) sheetからdownloadしたcsvから必要な情報を抽出する
    if args.do_csv_set:
        extract_sheet2csv(args)
    
    # (3) csvからjsonlファイルを作成する
    if args.do_all_json:
        make_jsonl(args)

    # (4) sentenceにmecabを適用
    if args.do_mecab_json:
        exex_mecab(args)
    
    # (5) jsonlからtrain/val/testに分けたjsonlをそれぞれ作成
    if args.do_split_json:
        create_split_jsonl(args)
    
    # (6) 評価caption用のjsonファイルの作成
    if args.do_caption_json:
        make_caption(args)
    
    # (7) 上位top-kのmax_t_lenを計算する
    if args.do_calc_t_len:
        calc_topk_length(args)


if __name__ == "__main__":
    main()
