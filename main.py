from pyparsing import Word, alphas, nums, oneOf, infixNotation, ZeroOrMore
import torch
import pysimplegui as sg

def create_nn(layer_defs):
    layers = []
    for defn in layer_defs:
        input_size, output_size = defn
        layer = torch.nn.Linear(input_size, output_size)
        layers.append(layer)
    return torch.nn.Sequential(*layers)

def parse_program(program):
    number = Word(nums).setName("number")
    variable = Word(alphas).setName("variable")
    comma = oneOf(",").setName(",")
    colon = oneOf(":").setName(":")
    open_bracket = oneOf("[").setName("[")
    close_bracket = oneOf("]").setName("]")
    nn_layers = open_bracket + number + comma + number + close_bracket
    nn_definition = "nn" + colon + nn_layers + ZeroOrMore(comma + nn_layers)
    result = nn_definition.parseString(program)
    return result[2:]

def create_gui(nn):
    layout = [
        [sg.Text("Enter a number:")],
        [sg.InputText(key="input")],
        [sg.Button("Send")],
        [sg.Text("Output:")],
        [sg.Output(key="output")],
    ]
    window = sg.Window("Neural Network", layout)
    return window

def main_loop(window, nn):
    while True:
        event, values = window.read()
        if event == "Send":
            try:
                input_message = float(values["input"])
                output_message = nn(torch.tensor([input_message]))
                print(output_message.item(), end="", file=window["output"])
            except ValueError:
                print("Invalid input. Please enter a number.", end="", file=window["output"])
        elif event == sg.WINDOW_CLOSED:
            break
    window.close()

def main():
    program = "nn: [64, 128], [128, 64], [64, 10]"
    layer_defs = parse_program(program)
    nn = create_nn(layer_defs)
    window = create_gui(nn)
    main_loop(window, nn)

if __name__ == "__main__":
    main()
