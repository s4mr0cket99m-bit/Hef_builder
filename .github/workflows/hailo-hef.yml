name: Compile DistilGPT2 to Hailo HEF

on:
  push:
    branches: [ main, master ]    # Run on pushes to main/master
  workflow_dispatch:             # Allow manual trigger from GitHub UI

jobs:
  compile_model:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4  # Updated from v3 to v4 [oai_citation:3‡cicube.io](https://cicube.io/workflow-hub/actions-setup-python/#:~:text=deploy%3A%20runs,run%3A%20python%20my_script.py)
      
      - name: Set up Python
        uses: actions/setup-python@v5  # Updated from v4 to v5 [oai_citation:4‡cicube.io](https://cicube.io/workflow-hub/actions-setup-python/#:~:text=deploy%3A%20runs,run%3A%20python%20my_script.py)
        with:
          python-version: '3.10'      # Use Python 3.10 (or any supported 3.x version)
      
      - name: Install Hailo Dataflow Compiler
        run: pip install "https://github.com/s4mr0cket99m-bit/Hef_builder/releases/download/dfc-1/hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64.whl"
        # Downloads and installs Hailo DataFlow Compiler v3.33.0 from the specified wheel
      
      # (Optional) If additional Python packages are needed (e.g., ONNX runtime or Hugging Face tools), install them here.
      # - name: Install additional requirements
      #   run: pip install transformers onnx
      
      - name: Compile DistilGPT2 model to HEF
        run: |
          # Assume 'distilgpt2.onnx' is available (in the repo or downloaded prior to this step).
          # 1. Optimize the ONNX model to a Hailo HAR (Hailo Archive) file for Hailo-8 device
          hailo optimize distilgpt2.onnx --output-har-path distilgpt2.har --hw-arch hailo8
          # 2. Compile the HAR file to a HEF file
          hailo compile distilgpt2.har --output-dir . --hw-arch hailo8
          # Note: The compiler will produce a .hef file in the current directory (name may derive from model or config)
      
      - name: Ensure consistent HEF filename
        run: |
          # Rename the generated .hef file to 'distilgpt2.hef' for consistency
          HEF_FILE=$(find . -maxdepth 1 -type f -name "*.hef" | head -n 1)
          if [ "$HEF_FILE" != "distilgpt2.hef" ]; then
            mv "$HEF_FILE" distilgpt2.hef
          fi
          ls -lh distilgpt2.hef  # List the file to confirm it exists with the correct name
      
      - name: Upload HEF artifact
        uses: actions/upload-artifact@v4  # Updated from v3 to v4 [oai_citation:5‡chsami.com](https://chsami.com/blog/upgrading-github-v4-artifacts/#:~:text=,app%20path%3A%20%24%7B%7Benv.DOTNET_ROOT%7D%7D%2Fmyapp)
        with:
          name: distilgpt2-hef   # Artifact name shown in GitHub UI
          path: distilgpt2.hef   # Path of the .hef file to upload
