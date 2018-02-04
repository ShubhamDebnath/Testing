# Learning to use object_detection api from tensorflow

#### This is basically me following SentDex's tutorials, 
#### You can probably get all the helper code from pythonprogramming.net

## Steps to follow

#### 1) Download protobuf-python compiler (preferably v3.4.0, need to change train.py in object_detection folder accordingly) from <a href = https://github.com/google/protobuf/releases/>Protobuf</a> and run these commands
###### set PYTHONPATH=$PYTHONPATH:`pwd`:`../models/research/slim'
######  "C:/Program Files/protoc/bin/protoc" object_detection/protos/*.proto --python_out=.

#### 2) Download <a href=https://github.com/tzutalin/labelImg>labelimg</a> and run these
###### pip install pyqt5
###### pyrcc4 -o resources.py resources.qrc
###### python labelImg.py

#### 3) Create xml files for all your pictures and put them in test/train folders

#### 4) Run these lines
###### python xml_to_csv.py
###### protoc object_detection/protos/*.proto --python_out=.
###### python3 setup.py install
###### python generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record
###### python generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record

#### 5) Download a model of your choice from <a href =https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md >here</a> and also take its related config file from <a href = https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs>here</a>, keep both in object_detection folder. You need to make a few changes in the config file, or you can just take <a href=https://github.com/ShubhamDebnath/Testing/blob/master/Object%20Detection/training/ssd_mobilenet_v1_coco.config>mine</a> from training folder

#### 6) Create a pbtext file an example of which is <a href = https://github.com/ShubhamDebnath/Testing/blob/master/Object%20Detection/training/duke_labels.pbtxt>this</a>, id referes to the label , start it with 1 , not 0

#### 7) Start training , depending on the model you chose, you may want to change that in this command
###### python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config

#### 8) Reap the fruits of your labour , by running 'object_detection_tutorial.ipynb' from jupyter notebook
###### jupyter notebook
