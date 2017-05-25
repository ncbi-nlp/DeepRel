#!/usr/bin/env bash
python cnn/preparse.py \
    ~/panfs/software/geniatagger-3.0.2/geniatagger \
    ~/panfs/software/stanford-corenlp-full-2016-10-31/ \
    ~/data/pengyifan-cnn/all \
    ~/data/pengyifan-cnn/train-foo.json \
    ~/data/pengyifan-cnn/test-foo.json \
    ~/data/pengyifan-cnn/dev-foo.json
python cnn/create_features.py \
    ~/data/pengyifan-cnn/all \
    ~/data/pengyifan-cnn/train-foo.json \
    ~/data/pengyifan-cnn/test-foo.json \
    ~/data/pengyifan-cnn/dev-foo.json
python cnn/create_vocabs.py \
    -w ~/data/word2vec/PubMed-and-PMC-w2v.bin \
    -e \
    -o ~/data/pengyifan-cnn/cnn_model/ \
    ~/data/pengyifan-cnn/all \
    ~/data/pengyifan-cnn/train-foo.json \
    ~/data/pengyifan-cnn/test-foo.json \
    ~/data/pengyifan-cnn/dev-foo.json
# matrix
python cnn/create_matrix.py \
    ~/data/pengyifan-cnn/cnn_model/vocabs.json \
    ~/data/pengyifan-cnn/all \
    ~/data/pengyifan-cnn/train-foo.json \
    ~/data/pengyifan-cnn/cnn_model/train-foo.npz
python cnn/create_matrix.py \
    ~/data/pengyifan-cnn/cnn_model/vocabs.json \
    ~/data/pengyifan-cnn/all \
    ~/data/pengyifan-cnn/test-foo.json \
    ~/data/pengyifan-cnn/cnn_model/test-foo.npz
python cnn/create_matrix.py \
    ~/data/pengyifan-cnn/cnn_model/vocabs.json \
    ~/data/pengyifan-cnn/all \
    ~/data/pengyifan-cnn/dev-foo.json \
    ~/data/pengyifan-cnn/cnn_model/dev-foo.npz
# shortest path matrix
python cnn/create_sp_matrix.py \
    ~/data/pengyifan-cnn/cnn_model/vocabs.json \
    ~/data/pengyifan-cnn/all \
    ~/data/pengyifan-cnn/train-foo.json \
    ~/data/pengyifan-cnn/cnn_model/train-foo-sp.npz
python cnn/create_sp_matrix.py \
    ~/data/pengyifan-cnn/cnn_model/vocabs.json \
    ~/data/pengyifan-cnn/all \
    ~/data/pengyifan-cnn/test-foo.json \
    ~/data/pengyifan-cnn/cnn_model/test-foo-sp.npz
python cnn/create_sp_matrix.py \
    ~/data/pengyifan-cnn/cnn_model/vocabs.json \
    ~/data/pengyifan-cnn/all \
    ~/data/pengyifan-cnn/dev-foo.json \
    ~/data/pengyifan-cnn/cnn_model/dev-foo-sp.npz
# doc2vec
python cnn/train_doc2vec.py \
    ~/data/pengyifan-cnn/all \
    ~/data/pengyifan-cnn/train-foo.json \
    ~/data/pengyifan-cnn/cnn_model/train-foo.doc2vec
python cnn/create_doc2vec.py \
    ~/data/pengyifan-cnn/all \
    ~/data/pengyifan-cnn/cnn_model/train-foo.doc2vec \
    ~/data/pengyifan-cnn/train-foo.json \
    ~/data/pengyifan-cnn/cnn_model/train-foo-doc.npz
python cnn/create_doc2vec.py \
    ~/data/pengyifan-cnn/all \
    ~/data/pengyifan-cnn/cnn_model/train-foo.doc2vec \
    ~/data/pengyifan-cnn/dev-foo.json \
    ~/data/pengyifan-cnn/cnn_model/dev-foo-doc.npz
python cnn/create_doc2vec.py \
    ~/data/pengyifan-cnn/all \
    ~/data/pengyifan-cnn/cnn_model/train-foo.doc2vec \
    ~/data/pengyifan-cnn/test-foo.json \
    ~/data/pengyifan-cnn/cnn_model/test-foo-doc.npz
# test matrix
python cnn/test_matrix.py \
    ~/data/pengyifan-cnn/cnn_model/vocabs.json \
    ~/data/pengyifan-cnn/cnn_model/train-foo-sp.npz \
    ~/data/pengyifan-cnn/cnn_model/dev-foo-sp.npz \
    ~/data/pengyifan-cnn/cnn_model/test-foo-sp.npz
# train
python cnn/train.py ~/data/pengyifan-cnn/cid.ini
# test
python cnn/test.py ~/data/pengyifan-cnn/cnn_model