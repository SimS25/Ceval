import numpy as np
import tensorflow as tf
import numpy as np
import math
import rectpack
from rectpack import newPacker

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
    # output = (function code, name of the output variable)
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
    ## Dense Layer data format:
    #           X
    #     ____________
    #     |           |
    # Y   |     W     |
    #     |___________|
    # Y+1 |____ B ____|
    #   X = dense layer OUTPUT dimension
    #   Y = dense layer INPUT dimension (usually output from the previous layer)
    #   B = Y+1 is bias row

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

    def get_embedded_data_str(self) -> str:
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

    # save coordinates of the data for loading and binarization
    def set_data_coordinates(self, xCoord, yCoord):
        self.data_coordinates = (xCoord, yCoord);

    # write output binary data into the prepared coordinates
    def write_data_binary(self, data):
        xCoord, yCoord = self.data_coordinates
        xDim, yDim = self.get_dimensions();

        # write weights
        for y in range(yDim):
            for x in range(xDim):
                data[yCoord + y][xCoord + x] = self.weights[y][x];
        # write biases
        for x in range(xDim):
            data[yCoord + yDim][xCoord + x] = self.biases[x];

    # return dimensions of the layer
    def get_dimensions(self) ->(int, int):
        return (len(self.weights[0]), len(self.weights));

    # return required memory space for this dense layer
    def get_memory(self):
        x,y = self.get_dimensions();
        return (x, y + 1); # y dimension in data will store additional row for bias

    # layer index
    def get_layer_index(self):
        return self.index;

    # returns data index into the dense layer data texture
    def _get_data_index(self, x : str, y : str) -> str:
        xOffset = "";
        if self.data_coordinates[0] > 0:
            xOffset = str(self.data_coordinates[0]) + " + ";
        yOffset = "";
        if self.data_coordinates[1] > 0:
            yOffset = str(self.data_coordinates[1]) + " + ";
        return "int3(" + xOffset + x + ", " + yOffset  + y + ", 0)";

    # input = x,y position in the weight data
    # output = str of the code loading the data into the position
    def _load_weight(self, x : str, y : str) -> str:
        if self.ceval.embedded:
            weight = self.var + "w" + str(self.index);
            return weight + "[" + y + "][" + x + "]"; # embedded data are sorted in transpose for better memory access
        else:
            return self.ceval.dense_data_name + ".Load(" + self._get_data_index(x, y) + ")";

    # input = x position in the bias data
    # output = str of the code loading the data into the position
    def _load_bias(self, x: str) -> str:
        if self.ceval.embedded:
            bias = self.var + "b" + str(self.index);
            return bias + "[" + x + "]";
        else:
            xDim, yDim = self.get_dimensions();
            return self.ceval.dense_data_name + ".Load(" + self._get_data_index(x, str(yDim)) + ")";

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
        inner_weight_loop = _CodeGenerator.for_loop("int o = 0; o < " + str(output_len) + "; o++", output + "[o] += " + self._load_weight("o", "i") + " * " + input + "[i];", "", indent);
        code = code + _CodeGenerator.for_loop("int i = 0; i < " + str(input_len) + "; i++", inner_weight_loop, indent, indent) + _CodeGenerator.new_line;

        # loop over biases
        code = code + _CodeGenerator.for_loop("int o = 0; o < " + str(output_len) + "; o++", output + "[o] = " + self.activation + "(" + output + "[o] + " + self._load_bias("o") + ");", indent, indent) + _CodeGenerator.new_line + _CodeGenerator.new_line;

        return (code, output);

# helper class for packing layer data
class _DataPacker:
    # pack rectangles into the canvas, returns None if canvas is too small
    def _pack_rectangles(rectangles, x, y, num_layers):
        packer = newPacker(rotation=False);
        # Add the rectangles to packing queue
        for r in rectangles:
            packer.add_rect(*r)

        # Add the bins where the rectangles will be placed
        bins = [(x, y)]
        packer.add_bin(*bins[0]);
        # Start packing
        packer.pack()
        # Obtain number of bins used for packing
        nbins = len(packer);

        if nbins == 1:
            if (len(packer[0]) < num_layers):  # not all rectanges fit into the bin
                return None
            else:
                return packer;
        else:
            return None;

    # dense layer comparator to sort layers from the largest to the smallest
    def _denseComparator(d):
        x, y = d.get_memory();
        return x * y;

    # pack layers by preparing their data index into common data canvas
    # data canvas resolution is returned on the output
    def pack_layers_data(dense_layers) ->(int, int):
        # sort layers by their dense layer size
        dense_layers.sort(reverse=True, key=_DataPacker._denseComparator);

        # start with the dimensions of the largest layer
        xSize, ySize = dense_layers[0].get_memory();

        # ceil them up to power of two
        powerX = math.log(xSize) / math.log(2.0);
        powerY = math.log(ySize) / math.log(2.0);
        xSize = int(math.pow(2.0, math.ceil(powerX)));
        ySize = int(math.pow(2.0, math.ceil(powerY)));

        # now try to fit blocks into the space, and increase size otherwise
        rectangles = [];
        for dense in dense_layers:
            x, y = dense.get_memory();
            rectangles.append((x, y, dense.get_layer_index()));

        final_packer = None;
        while True:
            # try to pack to the current size
            packer = _DataPacker._pack_rectangles(rectangles, xSize, ySize, len(dense_layers));
            if packer is not None:
                final_packer = packer;
                break;

            # try to enlarge just the xSize
            packer = _DataPacker._pack_rectangles(rectangles, 2 * xSize, ySize, len(dense_layers));
            if packer is not None:
                final_packer = packer;
                break;

            # try to enlarge just the ySize
            packer = _DataPacker._pack_rectangles(rectangles, xSize, 2 * ySize, len(dense_layers));
            if packer is not None:
                final_packer = packer;
                break;

            # enlarge both and try again
            xSize = 2 * xSize;
            ySize = 2 * ySize;

        # propagate data coordinates to the dense layers
        for dense in dense_layers:
            # find coordinates of the rectangle in the packer
            xCoord, yCoord = (0, 0);
            for rect in final_packer[0]:
                if rect.rid == dense.get_layer_index():
                    xCoord = rect.x;
                    yCoord = rect.y;
                    break;
            dense.set_data_coordinates(xCoord, yCoord);

        # final canvas size for the whole dense data
        xSize = final_packer.bin_list()[0][0];
        ySize = final_packer.bin_list()[0][1];
        return (xSize, ySize);

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

        function_header = "// prediction function" + _CodeGenerator.new_line;
        function_header = function_header + "void " + self.function_name + "(";
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
                self.dense_layers.append(self.layers[-1]); # save dense layer for further data processing
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

    # generate data embedded into the header file
    def __generate_embedded_data(self, file):
        for layer in self.layers:
            # write data of the layer
            file.write(layer.get_embedded_data_str());

    # generate binary data for the dense layers of the network
    # returns dimension of the texture
    def __generate_binary_dense_data(self) ->(int, int):
        # pack data
        xSize, ySize = _DataPacker.pack_layers_data(self.dense_layers);

        # create empty canvas of the size result from the 2D packing
        data = [[0 for col in range(xSize)] for row in range(ySize)]

        # write all layers in serial, filling up unused data space in between
        for dense in self.dense_layers:
            # write data into the block
            dense.write_data_binary(data);

        # write data to the file
        dense_data_file = open(self.dense_data_output_path, "wb");  # write bytes, always override previous content
        np_float_data = np.array(data, 'float32');
        np_float_data.tofile(dense_data_file);
        dense_data_file.close();

        return (xSize, ySize);

    # generate data of the network
    def __generate_data(self, file):
        # generate the binary data into separate files
        xDense, yDense = self.__generate_binary_dense_data();

        # prepare texture bind spots for user
        data_bind_definition = "// #TODO:" + _CodeGenerator.new_line;
        data_bind_definition = data_bind_definition + "//      1. Assign binding slots for the network data" + _CodeGenerator.new_line;
        data_bind_definition = data_bind_definition + "//      2. Load textures with their dimensions and bind them to their slots:" + _CodeGenerator.new_line;
        data_bind_definition = data_bind_definition + "//              a. Load " + self.dense_data_name + " as Texture2D with dimensions [" + str(xDense) + "][" + str(yDense) + "] and format float32" + _CodeGenerator.new_line;
        data_bind_definition = data_bind_definition + "//      3. Call function " + self.function_name + " to predict." + _CodeGenerator.new_line;
        data_bind_definition = data_bind_definition + "Texture2D<float> " + self.dense_data_name + " : register(t" + "100" + ");" + _CodeGenerator.new_line;
        file.write(data_bind_definition + _CodeGenerator.new_line);

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
        # dense layers structure name
        self.dense_data_name = self.function_name + "_dense_data";
        # dense data file output
        self.dense_data_output_path = output_folder + "/" + self.dense_data_name;
        # macro header
        self.macro_header = "__" + output.upper() + "__"
        # list for all hidden layers
        self.layers = [];
        # list for Dense layers
        self.dense_layers = [];
        # generate data as embedded into the header
        self.embedded = embedded;

    def generate(self, model : tf.keras.Model):
        # collect hidden layer data
        self.__gather_hidden_layers(model, self.function_name + "_");

        # open file for write the output
        file = open(self.output_path, "w");  # write, always override previous content

        # start by defining the macro
        self.__start_macro(file);

        # define data structures for the network
        if self.embedded:
            self.__generate_embedded_data(file); # embed all data directly into the header
        else:
            self.__generate_data(file); # define source texture to be used for data

        # write the output code
        self.__define_activation_functions(file);

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