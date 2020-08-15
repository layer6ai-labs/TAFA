<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

## RecSys'20 TAFA: Two-headed Attention Fused Autoencoder for Context-Aware Recommendations

Authors: Jinpeng Zhou*, Zhaoyue Cheng*, Felipe Perez, [Maksims Volkovs](http://www.cs.toronto.edu/~mvolkovs)

[[paper](http://www.cs.toronto.edu/~mvolkovs/recsys2020_tafa.pdf)]

<a name="Environment"/>

## Environment:

The code was developed and tested on the following python environment:
```
python 3.7.6
pytorch 1.4.0
pandas 1.1.0
tqdm
allennlp
gensim
cupy
nltk
```
<a name="instructions"/>

## Instructions:

To train and eveluate TAFA on the [Amazon datasets](http://jmcauley.ucsd.edu/data/amazon) (digital music, grocery and gourmet food, video games, cds and vinyl), run this command:
```bash
sh run_amazon.sh [digital_music|grocery_and_gourmet_food|video_games|cds_and_vinyl]
```

<a name="citation"/>

## Citation

If you find this code useful in your research, please cite the following paper:

    @inproceedings{zhou2020tafa,
      title={TAFA: {Two-headed} Attention Fused Autoencoder for Context-Aware Recommendations},
      author={Jinpeng Zhou, Zhaoyue Cheng, Felipe Perez, Maksims Volkovs},
      booktitle={RecSys},
      year={2020}
    }

