# Explore extreme compression for  pre-trained language models

Code for paper ["Exploring extreme parameter compression for pre-trained language models  ICLR2022"](https://openreview.net/forum?id=RftryyYyjiG) 

## Before Training

### install some libraries

``` 
 pip install tensorly==0.5.0
```

Torch is needed, torch 1.0-1.4 is preferred

Install horovod for distributed learning 

Configuration [Install horovod on GPU](https://github.com/horovod/horovod/blob/master/docs/gpus.rst)

```
pip install horovod[pytorch]
```

### Prepare corpus

```
cd data/corpus
wget -t 0 -c -T 20 https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
python WikiExtractor.py enwiki-latest-pages-articles.xml.bz2 -b 30G -q -o - > corpus.txt
```

### download pre-trained models

```bash
wget https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin -P  models/bert-base-uncased
wget https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt -P  models/bert-base-uncased
wget https://huggingface.co/bert-base-uncased/resolve/main/config.json -P  models/bert-base-uncased
cp models/bert-base-uncased/pytorch_model.bin models/bert-td-72-384/pytorch_model.bin 
cp models/bert-base-uncased/vocab.txt models/bert-td-72-384/vocab.txt
```



### generate training data for given  corpora (e.g., saved in the path "corpora" )

```
python pregenerate_training_data.py --train_corpus data/corpus \ 
                  --bert_model models/bert-base-uncased \
                  --do_lower_case \
                  --epochs_to_generate 3 \
                  --output_dir data/pregenerated_data
```

Due to the large size of the corpus, here we also offer a small pregenerated dataset for debugging.

## task data augmentation

```
python data_augmentation.py --pretrained_bert_model ${BERT_BASE_DIR}$ \
                            --glove_embs ${GLOVE_EMB}$ \
                            --glue_dir ${GLUE_DIR}$ \  
                            --task_name ${TASK_NAME}$
```



## Decomposing BERT 



###  decomposition and general distillation

Run with horovod

```c
mpirun -np 8 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python3 general_distill.py --teacher_model models/bert-base-uncased --student_model models/bert-gd-72-384 --pregenerated_data data/pregenerated_data --num_train_epochs 2.0 --train_batch_size 32 --output_dir output/bert-gd-72-384 -use_swap --do_lower_case
```

 To restrict sharing among SAN or FFN,  add "ops" and set "ops" to be "san" or "ffn" in `bert-gd-72-384/config.json`

```
ops = "san"
```



## Evaluation 

### Task distillation with data augmentation in fine-tuning phase

Rename a pretrained model as "", for instance, change `step_0_pytorch_model.bin` to `pytorch_model.bin`, and change `load_compressed_model` from false to true in `output/config.json`

Task distillation for distributed training

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 task_distill.py --teacher_model models/bert-base-uncasedi/STS-B --student_model models/bert-gd-72-384 --task_name STS-B --aug_train --data_dir data/glue_data/SST-2 --max_seq_length 128 --train_batch_size 32 --aug_train --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./output/36-256-STS-B
```

Task distillation for single gpu

```
python3  task_distill.py  --teacher_model models/bert-base-uncased   --student_model  models/bert-td-72-384  --output output_demo  --data_dir  data/glue_data/SST-2   --task_name  SST-2  --do_lower_case --aug_train   
```
For augmentation, you should add `--aug_train`

Get test result for model

```
python run_glue.py --model_name_or_path  models/bert-td-72-384/SST-2 --task_name SST-2 --do_eval --do_predict --data_dir data/glue_data/STS-B --max_seq_length 128 --save_steps 500 --save_total_limit 2 --output_dir ./output/SST-2
```



