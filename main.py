from pyparsing import Word, alphas, nums, oneOf, infixNotation
import torch
import pysimplegui as sg

# define the tokens for the language
number = Word(nums).setName("number")
variable = Word(alphas).setName("variable")
string = Word(alphanums + " ").setName("string")
comma = oneOf(",").setName(",")
colon = oneOf(":").setName(":")
open_bracket = oneOf("[").setName("[")
close_bracket = oneOf("]").setName("]")

# define the syntax for creating a neural network
nn_layers = open_bracket + number + comma + number + close_bracket
nn_definition = "nn" + colon + nn_layers + zeroOrMore(comma + nn_layers)

# define the syntax for training the neural network
nn_train = "train" + colon + variable

# parse the input program
result = nn_definition.parseString("nn: [64, 128], [128, 64], [64, 10]")

# extract the layer definitions from the parse result
layer_defs = result[2:]

# create a list of torch.nn.Linear layers
layers = []
for defn in layer_defs:
    input_size, output_size = defn
    layer = torch.nn.Linear(input_size, output_size)
    layers.append(layer)

# create the neural network using the layers
nn = torch.nn.Sequential(*layers)

# define the GUI layout
layout = [
    [sg.Text("Enter a message:")],
    [sg.InputText(key="input")],
    [sg.Button("Send")],
    [sg.Text("Output:")],
    [sg.Output(key="output")],
]

# create the GUI window
window = sg.Window("Chatbot", layout)

# define the main loop
while True:
    # get the next event from the GUI
    event, values = window.read()

    # check if the "Send" button was clicked
    if event == "Send":
        # get the input message from the GUI
        input_message = values["input"]

        # pass the input message through the neural network
        output_message = nn(torch.tensor(input_message))

        # display the output message in the GUI
        print(output_message, end="", file=window["output"])

# close the GUI window
window.close()
