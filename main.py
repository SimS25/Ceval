import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

import ceval

# network arguments
parser = argparse.ArgumentParser()
#parser.add_argument("--model", default="../TFModels/recolor_relu_2xH8", type=str, help="TF model input")
parser.add_argument("--model", default="../TFModels/edge_detector_relu_2xH32", type=str, help="TF model input")
parser.add_argument("--output_folder", default="../HLSLModels", type=str, help="Output folder")
parser.add_argument("--output", default="edge_detector", type=str, help="Model output name")
parser.add_argument("--lang", default="HLSL", type=str, help="Language output model")
parser.add_argument("--embedded", default=None, help="Embed network weights into the header. Only small networks can be embedded.")

def test_predict(model : tf.keras.Model):
    test_val = np.array([0.1, 0.5, 0.75]);
    test_val = test_val.reshape(1,3);
    test_val = tf.convert_to_tensor(test_val);

    test_out = model.predict(test_val);
    print(test_val)
    print(test_out);

def main(args: argparse.Namespace):
    # load model
    model = tf.keras.models.load_model(args.model);
    #test_predict(model); # model testing

    # generate model code
    cv = ceval.Ceval(args.output, args.output_folder, args.lang, args.embedded is not None)
    cv.generate(model);

if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
