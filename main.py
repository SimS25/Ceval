import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import tensorflow as tf

import ceval

# network arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="../TFModels/edge_detector_relu_2xH16", type=str, help="TF model input")
#parser.add_argument("--model", default="../TFModels/edge_detector_relu_2xH32", type=str, help="TF model input")
parser.add_argument("--output_folder", default="../HLSLModels", type=str, help="Output folder")
parser.add_argument("--output", default="edge_detector_norm", type=str, help="Model output name")
#parser.add_argument("--output", default="edge_detector", type=str, help="Model output name")
parser.add_argument("--lang", default="HLSL", type=str, help="Language output model")
parser.add_argument("--embedded", default=None, help="Embed network weights into the header. Only small networks can be embedded.")

def main(args: argparse.Namespace):
    # load model
    model = tf.keras.models.load_model(args.model);

    # generate model code
    cv = ceval.Ceval(args.output, args.output_folder, args.lang, args.embedded is not None)
    cv.generate(model);

if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
