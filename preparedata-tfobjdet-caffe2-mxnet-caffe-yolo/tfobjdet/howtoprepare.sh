# From tensorflow/models/research/


#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/home/majid/softwares/cudnn/8.0-7.0/lib64:$LD_LIBRARY_PATH
#export PYTHONPATH=$PYTHONPATH:/home/majid/Myprojects/ECCV18/models/research/slim:pwd:pwd/slim

object_detection/protos/protoc object_detection/protos/*.proto --python_out=.
# A GPU/screen config to run all jobs for training and evaluation in parallel.
# Execute:
# source /path/to/your/virtualenv/bin/activate
# screen -R TF -c all_jobs.screenrc

screen -t train 0 python train.py --train_log_dir=workdir/train
screen -t eval_train 1 python eval.py --split_name=train --train_log_dir=workdir/train --eval_log_dir=workdir/eval_train
screen -t eval_test 2 python eval.py --split_name=test --train_log_dir=workdir/train --eval_log_dir=workdir/eval_test
screen -t tensorboard 3 tensorboard --logdir=workdir


#wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
#tar -xvf VOCtrainval_11-May-2012.tar
python object_detection/dataset_tools/create_pascal_tf_record.py \
    --label_map_path=object_detection/data/pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2012 --set=train \
    --output_path=pascal_train.record



python object_detection/dataset_tools/create_pascal_tf_record.py \
    --label_map_path=object_detection/data/pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2012 --set=val \
    --output_path=pascal_val.record

#+data
#  -label_map file
#  -train TFRecord file
#  -eval TFRecord file
#+models
#  + model
#    -pipeline config file
#    +train
#    +eval
#ssd_mobilenet_v1_coco_2017_11_17
# changed batchsize from 24 to 1
export PATH_TO_YOUR_PIPELINE_CONFIG=object_detection/data/ssd_mobilenet_v1_coco.config
export PATH_TO_TRAIN_DIR=object_detection/data
python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --train_dir=${PATH_TO_TRAIN_DIR}
# or 
python train.py --logtostderr \
--train_dir=training/ \
--pipeline_config_path=training/ssd_mobilenet_v1_coco.config

export PATH_TO_EVAL_DIR=object_detection/data
# From the tensorflow/models/research/ directory
python object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --checkpoint_dir=${PATH_TO_TRAIN_DIR} \
    --eval_dir=${PATH_TO_EVAL_DIR}

python eval.py \
    --logtostderr \
    --pipeline_config_path=training/ssd_mobilenet_v1_coco.config \
    --checkpoint_dir=training/ \
    --eval_dir=eval/

export PATH_TO_MODEL_DIRECTORY=object_detection/models/model/
tensorboard --logdir=${PATH_TO_MODEL_DIRECTORY}



#To visualize the eval results
tensorboard --logdir=eval/

#TO visualize the training results
tensorboard --logdir=training/




# inference
python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/ssd_mobilenet_v1_coco.config \
    --trained_checkpoint_prefix training/model.ckpt-4350 \
    --output_directory pascal_trained_inference


python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training_dota/ssd_mobilenet_v1_coco.config \
    --trained_checkpoint_prefix training_dota/model.ckpt-1415 \
    --output_directory dota_trained_inference



############################# DOTA
source .profile
workon "tensorflow virtualenv"

export MAIN_DIR=/home/majid/Myprojects/ECCV18/models/research
export HOME_DIR=$MAIN_DIR:/object_detection
cd $HOME_DIR/object_detection_tensorflow/ssd_mobilenet_v1_coco_2017_11_17

# put data in VOCdevkit/VOC2007
export PYTHONPATH=$PYTHONPATH:$HOME_DIR:$MAIN_DIR:$MAIN_DIR/slim
#TODO change JPG to jpg
examples_path = os.path.join(data_dir, year, 'ImageSets', 'Main',
                 FLAGS.set + '.txt')
truncated.append(0)#int(obj['truncated'])
poses.append('top'.encode('utf8')) #obj['pose'].encode('utf8')
flags.DEFINE_string('label_map_path', 'data/dota_label_map.pbtxt',
                    'Path to label map proto')
  years = ['VOC2007']

python create_pascal_tf_record.py \
    --data_dir=VOCdevkit \
    --year=VOC2007 \
    --set=train \
    --output_path=dota_train.record

python create_pascal_tf_record.py \
        --data_dir=VOCdevkit \
        --year=VOC2007 \
        --set=val \
        --output_path=dota_val.record

#+data
#  -label_map file
#  -train TFRecord file
#  -eval TFRecord file
#+models
#  + model
#    -pipeline config file
#    +train
#    +eval
#ssd_mobilenet_v1_coco_2017_11_17
# changed batchsize from 24 to 1
python train.py --logtostderr \
--train_dir=training/ \
--pipeline_config_path=training/faster_rcnn_nas_coco.config

python eval.py \
    --logtostderr \
    --pipeline_config_path=training/faster_rcnn_nas_coco.config \
    --checkpoint_dir=training/ \
    --eval_dir=eval/

#To visualize the eval results
tensorboard --logdir=eval/

#TO visualize the training results
tensorboard --logdir=training/


# inference
python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/faster_rcnn_nas_coco.config \
    --trained_checkpoint_prefix training/model.ckpt-4350 \
    --output_directory pascal_trained_inference


python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training_dota/faster_rcnn_nas_coco.config \
    --trained_checkpoint_prefix training_dota/model.ckpt-1415 \
    --output_directory dota_trained_inference
