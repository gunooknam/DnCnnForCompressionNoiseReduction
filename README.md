# DnCnnForCompressionNoiseReduction
HEVC all Intra Compression Noise Reduction. (Simple example)

# train 

use pytorch(python)

* Train step
  * Data generation
    * set QP and Encoding and frame select. (encoding_data.py)
    * Split the data into patches. (createCompressNoisePatch.py)
    * convert the data to h5 file (createCompressNoisePatch.py)
  * Train Model
    * setting training configuration
    * train model (train_model.py)

# inference

use libtorch(C++)

* serialize pytorch weight file (trace_model.py)
* run trained model in c++ (main.cpp)