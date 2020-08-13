cd data

wget https://recsys2020-layer6ai.s3.ca-central-1.amazonaws.com/glove_word2vec.txt?versionId=o.Az..HNB.tHbBpIV8bwTKlaoeyqjfcb
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Digital_Music_5.json.gz
gzip reviews_Digital_Music_5.json.gz

cd ../

python preprocess/prepro_amazon.py data/Digital_Music_5.json data/amazon/ data/glove_word2vec.txt
