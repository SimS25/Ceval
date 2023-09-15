import numpy as np
import tensorflow as tf

# helper class for writing basic C-style code strings
class _CodeGenerator:
    # characters for new line
    new_line = "\n";

    # for loop
    def for_loop(iteration_string : str, inner_commands : str, start_indent : str, indent_size : str) -> str:
        code = start_indent + "for (" + iteration_string + ")" + _CodeGenerator.new_line;  # define cycle over input array
        code = code + start_indent + "{" + _CodeGenerator.new_line;  # define cycle over input array

        # indent inner commands
        indented_inner_commands = indent_size + inner_commands.replace(_CodeGenerator.new_line, _CodeGenerator.new_line + start_indent + indent_size);
        code = code + start_indent + indented_inner_commands + _CodeGenerator.new_line; # add indented inner commands
        code = code + start_indent + "}"  # close the weight loop
        return code;

# virtual layer class
class _Layer:
    # initializing constructor
    def __init__(self, ceval):
        self.ceval = ceval;

    # get data for the layer
    def get_embedded_data_str(self):
        return "";

    # input = name of the input variable
    # output = (fucntion code, name of the output variable)
    def get_code_str(self, input : str, indent : str) -> (str,str):
        return ("",input);

# flatten layer data
class _FlattenLayer(_Layer):
    # initializing constructor
    def __init__(self, ceval, layer, layer_index, input_shape):
        # index of the flatten layer
        self.index = layer_index
        # input shape to be flattened
        self._input_shape = input_shape
        # call layer constructor
        _Layer.__init__(self, ceval);

    # input = name of the input variable
    # output = (fucntion code, name of the output variable)
    def get_code_str(self, input: str, indent: str) -> (str, str):
        output = "f" + str(self.index);
        output_len = 1;
        for s in self._input_shape:
            output_len = output_len * s;

        # define output variable
        code = indent + "// flatten layer " + str(self.index) + _CodeGenerator.new_line;
        code = code + indent + "float " + output + "[" + str(output_len) + "];" + _CodeGenerator.new_line;

        # support only for 2D inputs for the flatten
        if len(self._input_shape) == 2:
            # loop over weights
            inner_loop = _CodeGenerator.for_loop("int j = 0; j < " + str(self._input_shape[1]) + "; j++", output + "[i * " + str(self._input_shape[0]) + " + j] = " + input + "[i][j];", "", indent);
            code = code + _CodeGenerator.for_loop("int i = 0; i < " + str(self._input_shape[0]) + "; i++", inner_loop, indent, indent) + _CodeGenerator.new_line + _CodeGenerator.new_line;

        return (code, output);

# dense layer data
class _DenseLayer(_Layer):
    # initializing constructor
    def __init__(self, ceval, layer, layer_index, variable_prefix : str, activation : str):
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
        # call layer constructor
        _Layer.__init__(self, ceval);

    def get_embedded_data_str(self):
        weights_data = "static const float " + self.var + "w"+str(self.index)+"["+str(len(self.weights))+"]"+"["+str(len(self.weights[0]))+"] = " + _CodeGenerator.new_line + "{";
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
                weights_data = weights_data + "," + _CodeGenerator.new_line;
        weights_data = weights_data + "};" + _CodeGenerator.new_line;
        biases_data = "static const float " + self.var + "b"+str(self.index)+"["+str(len(self.biases))+"] = " + _CodeGenerator.new_line + "{";
        for b in range(len(self.biases)):
            bias = self.biases[b];
            biases_data = biases_data + str(bias);
            if b < len(self.biases) - 1:
                biases_data = biases_data + "," + _CodeGenerator.new_line;
        biases_data = biases_data + "};" + _CodeGenerator.new_line;
        return weights_data + _CodeGenerator.new_line + biases_data + _CodeGenerator.new_line

    # input = x,y position in the weight data
    # output = str of the code loading the data into the position
    def _load_weight(self, x : str, y : str) -> str:
        if self.ceval.embedded:
            weight = self.var + "w" + str(self.index);
            return weight + "[" + x + "][" + y + "]";
        else:
            return "_load_weight !- NOT IMPLEMENTED";

    # input = x position in the bias data
    # output = str of the code loading the data into the position
    def _load_bias(self, x: str) -> str:
        if self.ceval.embedded:
            bias = self.var + "b" + str(self.index);
            return bias + "[" + x + "]";
        else:
            return "_load_bias !- NOT IMPLEMENTED";

    # input = name of the input variable
    # output = (function code, name of the output variable)
    def get_code_str(self, input : str, indent : str) -> (str,str):
        output = "d" + str(self.index);
        biases = self.var + "b" + str(self.index);
        output_len = len(self.biases);
        input_len = len(self.weights);

        # define output variable
        code = indent + "// dense layer " + str(self.index) + _CodeGenerator.new_line;
        code = code + indent + "float "+ output + "[" + str(output_len) + "] = {";
        for i in range(output_len):
            code = code + "0.0f";
            if i < output_len - 1:
                code = code + ", ";
        code = code + "};" + _CodeGenerator.new_line;

        # loop over weights
        inner_weight_loop = _CodeGenerator.for_loop("int o = 0; o < " + str(output_len) + "; o++", output + "[o] += " + self._load_weight("i","o") + " * " + input + "[i];", "", indent);
        code = code + _CodeGenerator.for_loop("int i = 0; i < " + str(input_len) + "; i++", inner_weight_loop, indent, indent) + _CodeGenerator.new_line;

        # loop over biases
        code = code + _CodeGenerator.for_loop("int o = 0; o < " + str(output_len) + "; o++", output + "[o] = " + self.activation + "(" + output + "[o] + " + self._load_bias("o") + ");", indent, indent) + _CodeGenerator.new_line + _CodeGenerator.new_line;

        return (code, output);

class Ceval:
    # network format
    format : str = "float";

    def __start_macro(self, file):
        file.write("#ifndef " + self.macro_header + _CodeGenerator.new_line);
        file.write("#define " + self.macro_header + _CodeGenerator.new_line + _CodeGenerator.new_line);

    def __end_macro(self, file):
        file.write(_CodeGenerator.new_line + "#endif" + _CodeGenerator.new_line);

    def __define_activation_functions(self, file):
        file.write("#ifndef __NN_ACTIVATIONS__" + _CodeGenerator.new_line);
        file.write("#define __NN_ACTIVATIONS__");
        linear = _CodeGenerator.new_line + "// linear activation" + _CodeGenerator.new_line + "float linearActivation(float x) { return x; }" + _CodeGenerator.new_line
        relu = _CodeGenerator.new_line + "// relu activation" + _CodeGenerator.new_line + "float reluActivation(float x) { return max(0.0f, x); }" + _CodeGenerator.new_line
        sigmoid = _CodeGenerator.new_line + "// sigmoid activation" + _CodeGenerator.new_line + "float sigmoidActivation(float x) { return 1.0f / (1.0f + exp(-x)); }" + _CodeGenerator.new_line
        file.write(linear + relu + sigmoid);
        file.write("#endif" + _CodeGenerator.new_line + _CodeGenerator.new_line);

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
        file.write(function_header + _CodeGenerator.new_line + "{" + _CodeGenerator.new_line);

    def __end_function(self, file):
        file.write(_CodeGenerator.new_line + "}" + _CodeGenerator.new_line);

    def __gather_hidden_layers(self, model, variable_prefix : str):
        config = model.get_config();
        flatten_index = 0;
        dense_index = 0;
        for l in range(len(model.layers)):
            layer = model.layers[l];
            layer_class = config["layers"][l + 1]["class_name"]
            if layer_class == "Dense":
                activation = config["layers"][l + 1]["config"].get("activation")
                self.layers.append(_DenseLayer(self, layer, dense_index, variable_prefix, activation));
                dense_index = dense_index + 1;
            elif layer_class == "Flatten":
                if config["layers"][l]["class_name"] == "InputLayer":
                    input_shape = config["layers"][l]["config"]["batch_input_shape"][1:];
                    self.layers.append(_FlattenLayer(self, layer, flatten_index, input_shape));
                    flatten_index = flatten_index + 1;
                else:
                    print("Flatten on non-InputLayer not supported");
            elif layer_class == "Dropout":
                continue

    def __generate_embedded_data(self, file):
        for layer in self.layers:
            # write data of the layer
            file.write(layer.get_embedded_data_str());

    def __generate_data(self, file):
        # all dense layers have a single texture they read out from
        # generate the texture header from the network name
        file.write(self.function_name + "_data")

    # initializing constructor
    def __init__(self, output, output_folder, lang, embedded):
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
        # generate data as embedded into the header
        self.embedded = embedded;

    def generate(self, model : tf.keras.Model):
        # collect hidden layer data
        self.__gather_hidden_layers(model, self.function_name + "_");

        # open file for write the output
        file = open(self.output_path, "w");  # write, always override previous content

        # start by defining the macro
        self.__start_macro(file);

        # write the output code
        self.__define_activation_functions(file);

        # define data structures for the network
        if self.embedded:
            self.__generate_embedded_data(file); # embed all data directly into the header
        else:
            self.__generate_data(file); # define source texture to be used for data

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
        output_code = indent + "// output layer" + _CodeGenerator.new_line;
        output_code = output_code + _CodeGenerator.for_loop("int o = 0; o < " + str(self.output_dimension) + "; o++", "output[o] = " + input_var_name + "[o];", indent, indent);
        file.write(output_code);

        # end function body
        self.__end_function(file);

        # end file by closing the macro
        self.__end_macro(file);

        file.close();