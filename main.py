import torch
import torch._dynamo as dynamo
import traceback


from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
#dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
dtype = torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Load and preprocess example images (replace with your own image paths)
image_names = ["examples/kitchen/images/" + str(f"{i:02}") + ".png" for i in range(0, 2)]
images = load_and_preprocess_images(image_names).to(device)

# Export the model to ONNX format
"""
print("\n" + "="*30 + " DEBUGGING GRAPH " + "="*30)
print("Attempting to capture the graph with dynamo.export...")
exp_program = None # Initialize to None
try:
    # Use dynamo.export which is closer to the ONNX export process
    # Setting aten_graph=True helps see the low-level operators
    exp_program, guards = dynamo.export(
        model,
        images, # Pass example inputs matching the export call
        aten_graph=True # Get the ATen level graph relevant to the error
    )

    # Corrected access: exp_program *is* the GraphModule
    print("\n--- Dynamo Exported Graph (ATen Graph) ---")
    exp_program.graph.print_tabular() # Corrected: removed .graph_module

    # print("\n--- Dynamo Exported Code ---")
    # print(exp_program.code) # Corrected: access code directly

    unsupported_instr = 'aten.rsub.Scalar'
    print(f"\n--- Searching for ${unsupported_instr} node ---")
    found_unsupported = False
    # Corrected access: iterate over exp_program.graph.nodes
    for node in exp_program.graph.nodes:
        # Check if it's a function call node and targets the problematic op
        if node.op == 'call_function' and str(node.target) == unsupported_instr:
             print(f"\n[FOUND NODE]: {node.format_node()}")
             # Try to print the stack trace associated with this node
             if node.stack_trace:
                 print("--- Stack Trace for Node ---")
                 print(node.stack_trace)
                 print("--------------------------")
             else:
                 print("(Stack trace not available for this node)")
             found_unsupported = True
             # You might want to break or continue searching after finding one
             # break

    if not found_unsupported:
        print(f"\n${unsupported_instr} node was NOT found in the manually traced graph.")
        print("The error might be occurring in ONNX-specific conversion passes after this graph generation.")
    else:
        print("\nFound the problematic node in the traced graph. Analyze the node and its stack trace above.")

except Exception as e:
    print(f"\nManual dynamo.export tracing failed: {e}")
    print(traceback.format_exc()) # Print full traceback for the manual trace failure
    print("\nProceeding with onnx.dynamo_export call anyway...")

print("\n" + "="*30 + " END DEBUGGING GRAPH " + "="*30 + "\n")
# === End of debugging section ===

"""
#

dynm = False
if not dynm:
    torch.onnx.export(model, images, "vggt.onnx")
else:
    export_options = torch.onnx.ExportOptions(dynamic_shapes=False)
    onnx_model = torch.onnx.dynamo_export(model, images, export_options=export_options)
    onnx_model.save("vggt.onnx")
print("Model exported to ONNX format.")

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)
