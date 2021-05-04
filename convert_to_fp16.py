import sys

import onnxmltools

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f'Syntax: {sys.argv[0]} <input> <output>')
        sys.exit()

    input_name = sys.argv[1]
    output_name = sys.argv[2]

    model = onnxmltools.load_model(input_name)

    model_fp16 = onnxmltools.utils.convert_float_to_float16(model)

    onnxmltools.save_model(model_fp16, output_name)

