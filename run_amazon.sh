#!/bin/bash

if [ $1 = "digital_music" ]; then
  echo "running tafa on amazon digital music dataset"

  rm -rf data
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
elif [ $1 = "grocery_and_gourmet_food" ]; then
  echo "running tafa on amazon grocery and gourmet food dataset"

  rm -rf data
  mkdir data
  cd data
  mkdir amazon

  wget https://recsys2020-layer6ai.s3.ca-central-1.amazonaws.com/glove_word2vec.txt?versionId=o.Az..HNB.tHbBpIV8bwTKlaoeyqjfcb -O glove_word2vec.txt
  wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Grocery_and_Gourmet_Food_5.json.gz
  gzip -d reviews_Grocery_and_Gourmet_Food_5.json.gz

  cd ../

  python preprocess/prepro_amazon.py ./data/reviews_Grocery_and_Gourmet_Food_5.json ./data/amazon/ ./data/glove_word2vec.txt

  python main.py --data_directory "./data/amazon/" \
                 --iteration 200 \
                 --attention_size 128 \
                 --attention_hidden_size 128 \
                 --dropout_p 0.5 \
                 --lam 1 \
                 --rank 200 \
                 --rec_batch_size 100 \
                 --root 1.2 \
                 --mode 'joint'
elif [ $1 = "video_games" ]; then
  echo "running tafa on amazon video games dataset"

  rm -rf data
  mkdir data
  cd data
  mkdir amazon

  wget https://recsys2020-layer6ai.s3.ca-central-1.amazonaws.com/glove_word2vec.txt?versionId=o.Az..HNB.tHbBpIV8bwTKlaoeyqjfcb -O glove_word2vec.txt
  wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games_5.json.gz
  gzip -d reviews_Video_Games_5.json.gz

  cd ../

  python preprocess/prepro_amazon.py ./data/reviews_Video_Games_5.json ./data/amazon/ ./data/glove_word2vec.txt

  python main.py --data_directory "./data/amazon/" \
                 --iteration 200 \
                 --attention_size 128 \
                 --attention_hidden_size 128 \
                 --dropout_p 0.25 \
                 --lam 1 \
                 --rank 200 \
                 --rec_batch_size 100 \
                 --root 1 \
                 --mode 'joint'
elif [ $1 = "cds_and_vinyl" ]; then
  echo "running tafa on amazon cds and vinyl dataset"

  rm -rf data
  mkdir data
  cd data
  mkdir amazon

  wget https://recsys2020-layer6ai.s3.ca-central-1.amazonaws.com/glove_word2vec.txt?versionId=o.Az..HNB.tHbBpIV8bwTKlaoeyqjfcb -O glove_word2vec.txt
  wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_5.json.gz
  gzip -d reviews_CDs_and_Vinyl_5.json.gz

  cd ../

  python preprocess/prepro_amazon.py ./data/reviews_CDs_and_Vinyl_5.json ./data/amazon/ ./data/glove_word2vec.txt

  python main.py --data_directory "./data/amazon/" \
                 --iteration 200 \
                 --attention_size 512 \
                 --attention_hidden_size 512 \
                 --dropout_p 0.25 \
                 --lam 0.001 \
                 --rank 500 \
                 --rec_batch_size 100 \
                 --root 1 \
                 --mode 'joint'
fi