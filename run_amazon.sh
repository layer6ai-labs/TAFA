mkdir data
cd data
mkdir amazon

wget https://recsys2020-layer6ai.s3.ca-central-1.amazonaws.com/glove_word2vec.txt?versionId=o.Az..HNB.tHbBpIV8bwTKlaoeyqjfcb -O glove_word2vec.txt
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Digital_Music_5.json.gz
gzip -d reviews_Digital_Music_5.json.gz

cd ../

python preprocess/prepro_amazon.py ./data/reviews_Digital_Music_5.json ./data/amazon/ ./data/glove_word2vec.txt

python main.py --data_directory "./data/amazon/" \
               --iteration 200 \
               --attention_size 256 \
               --attention_hidden_size 256 \
               --dropout_p 0.5 \
               --lam 1 \
               --rank 500 \
               --rec_batch_size 100 \
               --root 1 \
               --mode 'joint'
