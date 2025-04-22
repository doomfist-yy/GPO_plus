<<<<<<< HEAD
# GPO_plus
=======

We referred to the implementations of [VSE++](https://github.com/fartashf/vsepp) and [SCAN](https://github.com/kuanghuei/SCAN) to build up our codebase. 

## Training

Assuming the data root is ```/tmp/data```, we provide example training scripts for:

BUTD Region feature for the image feature, BERT-base for the text feature. See ```train_region.sh```

## Evaluation

Run ```eval.py``` to evaluate specified models on either COCO and Flickr30K. For evaluting pre-trained models on COCO, use the following command (assuming there are 4 GPUs, and the local data path is /tmp/data):

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 eval.py --dataset coco --data_path /tmp/data/coco
```

For evaluting pre-trained models on Flickr-30K, use the command: 

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 eval.py --dataset f30k --data_path /tmp/data/f30k
```

For evaluating pre-trained COCO models on the CxC dataset, use the command:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 eval.py --dataset coco --data_path /tmp/data/coco --evaluate_cxc
```


For evaluating two-model ensemble, first run single-model evaluation commands above with the argument ```--save_results```, and then use ```eval_ensemble.py``` to get the results (need to manually specify the paths to the saved results). 



>>>>>>> 36b842f (the first commit)
