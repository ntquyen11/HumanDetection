# set the number of max_batches - min 2000 per class:
max_batch=6000
# calculate the 2 steps values:
step1 = 0.8 * max_batch
step2 = 0.9 * max_batch
import re
# we also need to adjust the number of classes and a parameter called filter size 
# that are both is inside the model structure

# num_classes = len(labels)
# num_filters = (num_classes + 5) * 3
num_filters = (1 + 5) * 3
num_classes=1

#cfg_file = '/content/yolotinyv3_medmask_demo/yolov3-tiny_obj.cfg'
cfg_file = '/home/ntquyen/ntquyen/Machine_Learning/ThayHoang/HumanDetection/withDarknet/yolotinyv3_medmask_demo/yolov4-tiny.cfg'

with open(cfg_file) as f:
    s = f.read()
# (re.sub('[a-z]*@', 'ABC@', s))
s = re.sub('max_batches = \d*','max_batches = '+str(max_batch),s)
s = re.sub('steps=\d*,\d*','steps='+"{:.0f}".format(step1)+','+"{:.0f}".format(step2),s)
s = re.sub('classes=\d*','classes='+str(num_classes),s)
s = re.sub('pad=1\nfilters=\d*','pad=1\nfilters='+"{:.0f}".format(num_filters),s)
# pad=1\nfilters=\d\d
# s = re.sub('CUDNN=0','CUDNN=1',s)
# s = re.sub('OPENCV=0','OPENCV=1',s)

with open(cfg_file, 'w') as f:
  # s = re.sub('GPU=0','GPU=1',s)
  f.write(s)
