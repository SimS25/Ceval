import numpy as np
import tensorflow as tf

# virtual layer class
class _Layer:
    # get data for the layer
    def get_data_str(self):
        return "";

    # input = name of the input variable
    # output = (fucntion code, name of the output variable)
    def get_code_str(self, input : str, indent : str) -> (str,str):
        return ("",input);

# flatten layer data
class _FlattenLayer(_Layer):
    # initializing constructor
    def __init__(self, layer, layer_index, input_shape):
        # index of the flatten layer
        self.index = layer_index
        # input shape to be flattened
        self._input_shape = input_shape

    # input = name of the input variable
    # output = (fucntion code, name of the output variable)
    def get_code_str(self, input: str, indent: str) -> (str, str):
        output = "f" + str(self.index);
        output_len = 1;
        for s in self._input_shape:
            output_len = output_len * s;

        # define output variable
        code = indent + "float " + output + "[" + str(output_len) + "];\n";

        # support only for 2D inputs for the flatten
        if len(self._input_shape) == 2:
            # loop over weights
            code = code + indent + "for (int i = 0; i < " + str(self._input_shape[0]) + "; i++)\n" + indent + "{\n";  # define cycle over input array
            code = code + indent + indent + "for (int j = 0; j < " + str(self._input_shape[1]) + "; j++)\n" + indent + indent + "{\n";  # define cycle over output
            code = code + indent + indent + indent + output + "[i * " + str(self._input_shape[0]) + " + j] = " + input + "[i][j];\n"
            code = code + indent + indent + "}\n"  # close the inner loop
            code = code + indent + "}\n"  # close the weight loop

        return (code, output);

# dense layer data
class _DenseLayer(_Layer):
    # initializing constructor
    def __init__(self, layer, layer_index, variable_prefix : str, activation : str):
        layer_data = layer.get_weights()
        # index of the hidden layer
        self.index = layer_index
        # weights of the layer
        self.weights = layer_data[0]
        # biases of the layer
        self.biases = layer_data[1]
        # save activation function
        self.activation = activation + "Activation";
        # prefix name of each variable
        self.var = variable_prefix;

    def get_data_str(self):
        weights_data = "static const float " + self.var + "w"+str(self.index)+"["+str(len(self.weights))+"]"+"["+str(len(self.weights[0]))+"] = \n{";
        for r in range(len(self.weights)):
            row = self.weights[r];
            weights_data = weights_data + "{";
            for w in range(len(row)):
                weight = row[w];
                weights_data = weights_data + str(weight);
                if w < len(row) - 1:
                    weights_data = weights_data + ", ";
            weights_data = weights_data + "}";
            if r < len(self.weights) - 1:
                weights_data = weights_data + ",\n";
        weights_data = weights_data + "};\n";
        biases_data = "static const float " + self.var + "b"+str(self.index)+"["+str(len(self.biases))+"] = \n{";
        for b in range(len(self.biases)):
            bias = self.biases[b];
            biases_data = biases_data + str(bias);
            if b < len(self.biases) - 1:
                biases_data = biases_data + ",\n";
        biases_data = biases_data + "};\n";
        return weights_data + "\n" + biases_data + "\n"

    # input = name of the input variable
    # output = (fucntion code, name of the output variable)
    def get_code_str(self, input : str, indent : str) -> (str,str):
        output = "d" + str(self.index);
        weights = self.var + "w" + str(self.index);
        biases = self.var + "b" + str(self.index);
        output_len = len(self.biases);
        input_len = len(self.weights);

        # define output variable
        code = indent + "float "+ output + "[" + str(output_len) + "] = {";
        for i in range(output_len):
            code = code + "0.0f";
            if i < output_len - 1:
                code = code + ", ";
        code = code + "};\n";

        # loop over weights
        code = code + indent + "for (int i = 0; i < " + str(input_len) + "; i++)\n" + indent + "{\n"; # define cycle over input array
        code = code + indent + indent + "for (int o = 0; o < " + str(output_len) + "; o++)\n" + indent + indent + "{\n"; # define cycle over output
        code = code + indent + indent + indent + output + "[o] += " + weights + "[i][o] * " + input + "[i];\n"
        code = code + indent + indent + "}\n" # close the inner loop
        code = code + indent + "}\n" # close the weight loop

        # loop over biases
        code = code + indent + "for (int o = 0; o < " + str(output_len) + "; o++)\n" + indent + "{\n";
        code = code + indent + indent + output + "[o] = " + self.activation + "(" + output + "[o] + " + biases + "[o]);\n"
        code = code + indent + "}\n\n" # close the biases loop

        return (code, output);

class CodeGenerator:
    # network format
    format : str = "float";

    def __start_macro(self, file):
        file.write("#ifndef " + self.macro_header + "\n");
        file.write("#define " + self.macro_header + "\n\n");

    def __end_macro(self, file):
        file.write("\n#endif\n");

    def __define_activation_functions(self, file):
        file.write("#ifndef __NN_ACTIVATIONS__\n");
        file.write("#define __NN_ACTIVATIONS__");
        linear = "\n// linear activation\nfloat linearActivation(float x) { return x; }\n"
        relu = "\n// relu activation\nfloat reluActivation(float x) { return max(0.0f, x); }\n"
        sigmoid = "\n// sigmoid activation\nfloat sigmoidActivation(float x) { return 1.0f / (1.0f + exp(-x)); }\n"
        file.write(linear + relu + sigmoid);
        file.write("#endif\n\n");

    def __start_function(self, model, file):
        # define network input and output type
        config = model.get_config();
        self.input_dimension = config["layers"][0]["config"]["batch_input_shape"][1:];  # first dimension is empty
        self.output_dimension = config["layers"][-1]["config"]["units"];
        function_header = "void " + self.function_name + "(";
        # add input and output parameters
        function_header = function_header + "in "+self.format+" input";
        for d in range(len(self.input_dimension)):
            function_header = function_header + "["+str(self.input_dimension[d])+"]"

        function_header = function_header + ", ";
        function_header = function_header + "out "+self.format+" output[" + str(self.output_dimension) + "])";
        # write function header and open body
        file.write(function_header + "\n{\n");

    def __end_function(self, file):
        file.write("\n}\n");

    def __gather_hidden_layers(self, model, variable_prefix : str):
        config = model.get_config();
        for l in range(len(model.layers)):
            layer = model.layers[l];
            layer_class = config["layers"][l + 1]["class_name"]
            if layer_class == "Dense":
                activation = config["layers"][l + 1]["config"].get("activation")
                self.layers.append(_DenseLayer(layer, l, variable_prefix, activation));
            elif layer_class == "Flatten":
                if config["layers"][l]["class_name"] == "InputLayer":
                    input_shape = config["layers"][l]["config"]["batch_input_shape"][1:];
                    self.layers.append(_FlattenLayer(layer, l, input_shape));
                else:
                    print("Flatten on non-InputLayer not supported");
            elif layer_class == "Dropout":
                continue

    def __generate_embedded_data(self, file):
        for layer in self.layers:
            # write data of the layer
            file.write(layer.get_data_str());

    # initializing constructor
    def __init__(self, output, output_folder, lang):
        if lang == 'hlsl':
            suffix = ".h";
        else:
            suffix = ".h";

        # output path of the convertor
        self.output_path = output_folder + "/" + output + suffix;
        # function name
        self.function_name = output;
        # macro header
        self.macro_header = "__" + output.upper() + "__"
        # list for hidden layers
        self.layers = [];

    def generate(self, model : tf.keras.Model):
        # collect hidden layer data
        self.__gather_hidden_layers(model, self.function_name + "_");

        # open file for write the output
        file = open(self.output_path, "w");  # write, always override previous content

        # start by defining the macro
        self.__start_macro(file);

        # write the output code
        self.__define_activation_functions(file);

        # define embedded data structures for the network
        self.__generate_embedded_data(file);

        # start function
        self.__start_function(model, file);

        # implement network predict code
        indent = "    ";
        input_var_name = "input";
        for layer in self.layers:
            (code, output) = layer.get_code_str(input_var_name, indent);
            input_var_name = output;
            file.write(code);

        # assign the last output to the function output and return
        output_code = indent + "for (int o = 0; o < " + str(self.output_dimension) + "; o++)\n" + indent + "{\n";
        output_code = output_code + indent + indent + "output[o] = " + input_var_name + "[o];\n" + indent + "}"
        file.write(output_code);

        # end function body
        self.__end_function(file);

        # end file by closing the macro
        self.__end_macro(file);

        file.close();