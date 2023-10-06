# RelationaFutureCaptioningModel (RFCM)

<p align="center"><img src="assets/eye-catch.jpg" alt="Eye-catch" title="eye-catch" /></p>

This code implements the Relational Future Captioning Model (RFCM). The implementation is based on [COOT](https://github.com/gingsi/coot-videotext)

## Installation

We use `python=3.8.5` and `pytorch=1.7.1`. Tested on `Ubuntu 18.04` and `CUDA 10.1`.

~~~bash
git clone https://github.com/keio-smilab22/RelationalFutureCaptioningModel.git
cd RelationalFutureCaptioningModel
pip install -r requirements.txt
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
python nltk_setup.py
~~~
If you have problems with the `pycocoevalcap` package try uninstalling it and installing it with this command instead: `pip install git+https://github.com/salaniz/pycocoevalcap`
The METEOR metric requires `java`. Either install the latest Java 1.8 through the system (Tested with `Java RE 1.8.0_261`). Make sure your locale is set correct i.e. `echo $LANG` outputs `en_US.UTF-8`

## Download and extract for YooCook2-FC dataset
[COOT output Embeddings](https://drive.google.com/file/d/1atbI9HaFArNPeZzkvrJ9TnkCAal6gyUQ/view?usp=sharing) ~230mb, [Pretrained Captioning models](https://drive.google.com/file/d/1IV85_DXWx1SJL9ZJuT6Qvvyx8obE9f9x/view?usp=sharing) ~540 mb [YouCook2-FC dataset](https://drive.google.com/file/d/1NHs-C_CpxFN6tbMuwo_hNp_oFeOWlBqW/view?usp=drive_link) ~15 gb

~~~bash
tar -xzvf provided_embeddings.tar.gz
tar -xzvf provided_models_caption.tar.gz
tar -xzvf youcook2-fc.tar.gz
cp -r data/youcook2_next data/youcook2
mkdir experiments
mkdir experiments/caption
~~~

## Precompute all the text features

~~~bash
python data_read_youcook2_meta.py
python precompute_text.py youcook2 --cuda
~~~

### Rebuild MART cache

Download [glove](https://drive.google.com/file/d/1G-arTgUgfqTRKkui-euPZFrPK63mMYPA/view?usp=sharing)
~~~bash
mkdir pretrained_models
mv glove* pretrained_models/
python mart_build_vocab.py youcook2_next
~~~


### Train and validate RFCM on COOT embeddings

~~~bash
# YouCook2-FC
# Train RFCM on COOT clip embeddings and show results
./rfcm.sh

~~~
