#!/usr/bin/env python3
"""
Export ONNX dense neural network weights to a C header file for deployment on microcontrollers 
without any inference library dependency.

Notes:
 - Only supports fully-connected (dense) layers with Tanh activations and no activation on the
   output layer. This matches the architecture produced by build_mlp() in ppo_trainer.py.
 - Does not support conv layers, LSTM, attention, ReLU, sigmoid, etc. Recommend using LiteRT for
   Microcontrollers (formerly TensorFlow Lite Micro) for these (and other more complex 
   architectures).

Usage:
    python onnx_actor_to_c.py <onnx_path> [output_path]

Args:
    onnx_path: path to the exported actor.onnx file
    output_path: path to write the .h file (default: actor.h in the same directory as the ONNX file)

Example:
    python onnx_actor_to_c.py runs/BalanceBot-v0.../actor.onnx firmware/actor.h
"""

# Standard libraries
from pathlib import Path
import sys

# Third-party libraries
import onnx
import numpy as np


#-------------------------------------------------------------------------------
# Private functions

def _array_to_c(arr, name):
    """Convert numpy array to C float array declaration."""
    flat = arr.flatten().astype(np.float32)
    values = ", ".join(f"{v:.8f}f" for v in flat)
    shape = " x ".join(str(s) for s in arr.shape)

    return (
        f"// Shape: {shape}\n"
        f"static const float {name}[{len(flat)}] = {{\n"
        f"    {values}\n"
        f"}};\n"
    )

#-------------------------------------------------------------------------------
# Public functions

def export_onnx_actor_to_c(onnx_path, output_path=None):
    """
    Export an ONNX actor network to a self-contained C header file. Includes 
    weights, biases, dimension constants, and a forward pass function. No 
    inference library or runtime required.

    Args:
        onnx_path (Path): path to actor.onnx
        output_path (Path): path to write actor.h
    """
    # Construct output path
    onnx_path = Path(onnx_path)
    if output_path is None:
        output_path = onnx_path.parent / "actor.h"
    output_path = Path(output_path)

    # Load ONNX and extract weights
    model = onnx.load(str(onnx_path))
    weights = {}
    for initializer in model.graph.initializer:
        weights[initializer.name] = onnx.numpy_helper.to_array(initializer)

    # Print the weights
    print("Weights found:")
    for name, arr in weights.items():
        print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")

    # Collect layers in order
    layer_num  = 0
    layer_info = []
    layer_lines = []

    # Construct the layers
    for name, arr in weights.items():
        if len(arr.shape) == 2:  # weight matrix
            layer_lines.append(_array_to_c(arr, f"layer{layer_num}_weight"))
            layer_info.append((arr.shape[1], arr.shape[0]))
        elif len(arr.shape) == 1:
            layer_lines.append(_array_to_c(arr, f"layer{layer_num}_bias"))
            layer_num += 1

    # Remember the maximum hidden layer size
    max_hidden = max(s[1] for s in layer_info)

    # Build function comment and declaration
    fwd = []
    fwd.append("/**")
    fwd.append(" * DNN forward pass.")
    fwd.append(f" * obs:    input array  [{layer_info[0][0]}] — [pitch, pitch_rate, wheel_vel_left, wheel_vel_right]")
    fwd.append(f" * output: output array [{layer_info[-1][1]}] — [left_motor, right_motor], range approx [-1, 1]")
    fwd.append(" *")
    fwd.append(" * Note: clamp output to [-1, 1] before sending to motor driver.")
    fwd.append(" */")
    fwd.append("static inline void actor_forward(const float* obs, float* output) {")
    fwd.append(f"    float buf0[{max_hidden}];")
    fwd.append(f"    float buf1[{max_hidden}];")
    fwd.append(f"    const float* in  = obs;")
    fwd.append(f"    float*       out = buf0;")
    fwd.append("")

    # Build forward pass math
    for i, (in_size, out_size) in enumerate(layer_info):
        is_last = (i == layer_num - 1)
        activation = "none (output layer)" if is_last else "tanh"
        fwd.append(f"    // Layer {i}: {in_size} -> {out_size}, activation={activation}")
        fwd.append(f"    for (int o = 0; o < {out_size}; o++) {{")
        fwd.append(f"        float acc = layer{i}_bias[o];")
        fwd.append(f"        for (int j = 0; j < {in_size}; j++)")
        fwd.append(f"            acc += layer{i}_weight[o * {in_size} + j] * in[j];")
        if not is_last:
            fwd.append(f"        out[o] = tanhf(acc);")
        else:
            fwd.append(f"        out[o] = acc;")
        fwd.append(f"    }}")
        if i < layer_num - 1:
            fwd.append(f"    in  = out;")
            fwd.append(f"    out = (out == buf0) ? buf1 : buf0;")
        else:
            fwd.append(f"    for (int o = 0; o < {out_size}; o++)")
            fwd.append(f"        output[o] = out[o];")
        fwd.append("")
    fwd.append("}")

    # Build header
    lines = [
        "// DO NOT EDIT: Auto-generated actor network weights",
        f"// Source: {onnx_path.name}",
        "// Generated by onnx_dnn_to_c.py",
        "//",
        "// Usage (ESP32 Arduino):",
        "//   #include \"actor.h\"",
        "//   float obs[4]    = {pitch, pitch_rate, wheel_vel_l, wheel_vel_r};",
        "//   float action[2];",
        "//   actor_forward(obs, action);",
        "//   // clamp to [-1, 1] before sending to motors",
        "",
        "#pragma once",
        "#include <math.h>  // for tanhf()",
        "",
        f"#define ACTOR_OBS_SIZE    {layer_info[0][0]}",
        f"#define ACTOR_ACTION_SIZE {layer_info[-1][1]}",
        f"#define ACTOR_NUM_LAYERS  {layer_num}",
        "",
    ]
    lines += layer_lines
    lines += fwd

    # Write the giant string out to a file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))

    # Show what we've done
    print(f"C header written to {output_path}")
    print(f"  Layers:  {layer_num}")
    print(f"  Obs:     {layer_info[0][0]}")
    print(f"  Actions: {layer_info[-1][1]}")

    return output_path

#-------------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    # Make sure we have only 2 arguments
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(__doc__)
        sys.exit(1)

    # Get paths
    onnx_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) == 3 else None

    # Ensure onnx file exists
    if not onnx_path.exists():
        print(f"Error: {onnx_path} not found")
        sys.exit(1)

    # Do the export
    export_onnx_actor_to_c(onnx_path, output_path)