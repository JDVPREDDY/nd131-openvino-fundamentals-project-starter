# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

I have implemented the project using 2 pre-trained models.
1. person-detection-retail-0013
2. SSD_mobilenet_v2_coco from this (link)[http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz]

The first implementation worked well when FP16 was used. It was Good because the model was exclusively trained on pedestrian and persons dataset. The path to model xml file is 
```intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml```

The second implementation has some issues as the model was trained on a dataset which has diverse classes other than persons exclusively. The bounding box goes on and off in between and whenever it goes on a person is detected. The path to its xml file is:

```ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml```

## Explaining Custom Layers

The process behind converting custom layers involves regesterig cumstom layers as extensions to the model optimizer. We can allow to calculate the output of these custom layers in the original framework. For caffe model we may also need to specify the output size.

Some of the potential reasons for handling custom layers are that are not included into a list of known layers. If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was is very little as the optimizer ensures that there is least reduction in accuracy

The size of the model pre- and post-conversion was reduced, as they undergo quantization in the optimizer, the space required is reduced.

The inference time of the model pre- and post-conversion was improved. Pre-conversion model has very high inference time, on my PC which is Intel i3, it took atleasst 10 seconds to process a frame and the video playback is very very slow. After conversion, the inference time is improved.

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
  - (source)[https://github.com/amoussawi/caffe]
  - There was a problem with the prototxt file.
  - The issue can be sorted out if the caffe is installed. But since I dont have caffe on my system, I tred installing it on workspace but in vain.
  - So I have shifted to a tensorflow model as stated above.