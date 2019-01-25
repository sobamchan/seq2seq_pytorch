P=~/mnt/DATA/small_parallel_enja
CUDA_VISIBLE=2

python main.py\
    -use_cuda\
    -use_tensorboard\
    -corpus_name small_enja\
    -train_src $P/train.ja\
    -train_tgt $P/train.en\
    -valid_src $P/dev.ja\
    -valid_tgt $P/dev.en\
    -save_dir test\
    -src_voc_size 5000\
    -tgt_voc_size 5000\
    -model_name test\
    -attn_model general\
    -hid_n 256\
    -encoder_layers_n 3\
    -decoder_layers_n 3\
    -dropout 0.5\
    -epoch 10\
    -print_every 1
