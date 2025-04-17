#!/bin/bash
all_params=("$@")
echo "entry"
export PATH="/workspace/:$PATH"
export PATH="/workspace/Vina-GPU-2.1-main/AutoDock-Vina-GPU-2.1:$PATH"
mv vina_1.2.5_linux_x86_64 vina
cd /workspace/Evaluation
/bin/bash ${all_params}
