# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

I have implemented the project using the following pre-trained model from intel as it is giving more accurate results than others which I have tried.
1. person-detection-retail-0013

- The commands I have used to get this model are as follows:
    ``` cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader```
    Then,
    ```python downloader.py --name human-pose-estimation-0001 -o /home/workspace```
    
- This implementation worked well when FP16 was used. It was Good because the model was exclusively trained on pedestrian and persons dataset. The command used to run the final application is as follows:
``` python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

## Explaining Custom Layers

The process behind converting custom layers involves regesterig cumstom layers as extensions to the model optimizer. We can allow to calculate the output of these custom layers in the original framework. For caffe model we may also need to specify the output size.

Some of the potential reasons for handling custom layers are that are not included into a list of known layers. If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

Since models in IR (person-detection-retail-0013) can't be run directly on a PC and their pre-converted forms are unknown, I had run models -2,3 on my PC and compared their performances pre and post conversion.

- Accuracy: The accuracy is slghtly reduced after conversion to IR, it can be clearly understood by jitter. The pre-conversion model has less jitter than the post-conversion(IR) model. And sometimes post-conversion doesn't even give bounding box results. Still the performance of the post-converted model is okay as the model optimizer ensures that there is least reduction in accuracy.


- Size: The size of the model in terms of the memory it occupies is reduced after converting to IR. The pre-conversion model is heavy in terms of memory occupied. This is the main goal of optimizer to reduce the usage of resources on edge devices.

- Inference Time: Pre-conversion model has very high inference time, on my PC which is Intel i3, it took atleasst 10 seconds to process a frame and the video playback is very very slow. After conversion, the inference time is improved. The goal is to reduce the inference time to make the model run faster on edge devices which is achieved here (post-conversion)

## Assess Model Use Cases

Some of the potential use cases of the people counter app are:
1. It can be used to count people entering a lecture hall
2. It can also be used to count the no of people in a queue in front a gate/counter.

Each of these use cases would be useful because:
1. To keep track of no of people attending lectures.
2. To know the no of people in a queue at a given time and improve the speed of servoces offered.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows...

1. The model accuracy is slightly affected. But this should not affect the user needs for this appliation as this  is not something like a Self driving car where some wrong predictions may lead to catastrophic outcomes.
2. Lighting of the input feed may vary but it ultimately depends on the training data for that model. The training data should include both light and dark scenes so that model can learn something. So, choose models that are trained on diverse set of data.
3. Again Image size issues cn be there if image pyramids are not used while training model. Again training data also matters to include such images sizes.

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:
- Model 1: [Single Shot Multibox Detector on Caltech pedestrian dataset]
  - [source](https://github.com/amoussawi/caffe)
  - There was a problem with the prototxt file.
  - The issue can be sorted out if the caffe is installed. But since I dont have caffe on my system, I tred installing it on workspace but in vain.
  - So I have shifted to a tensorflow model as stated above.
  
- Model 2: [ssd_resnet50_v1_fpn_shared_box_predictor]
  - [source](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
  - I have used the following commands:
      - To download the model:
      ```wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz```
      - To unzip the downloaded file:
      ``` tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz```
      - Navigate to the directory:
      ``` cd ssd_mobilenet_v2_coco_2018_03_29```
      - To convert the model to IR:
      ```
      python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
      ```
      - The convetred xml file is in the specified path below:
      ``` ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml ```
      
  - This implementation has some issues as the model was trained on a dataset which has diverse classes other than persons exclusively. The bounding box goes on and off in between and whenever it goes on a person is detected. {say, jitter}
  - I have tried to fix the problem with skipping some frames in between but it doesn't seem like the optimal implementation. So, I have decided no to go with this.

- Model 3: [ssd_resnet50_v1_fpn_shared_box_predictor]
  - [source](http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz)
  - I have used the following commands:
      - To download the model:
      ```wget http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz```
      - To unzip the downloaded file:
      ``` tar -xvf ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz```
      - Navigate to the directory:
      ``` cd ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/```
      - To convert the model to IR:
      ```
      python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
      ```
      - The convetred xml file is in the specified path below:
      ```ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.xml ```
      
  - The problem with this model is jitter (same as with the model-2) and this is somewhat heavy model (in layers and parameters) as compared to previous ones :) 


  