#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=share-short
#PJM -L gpu=4
#PJM -N waveglow
#PJM -j
#PJM -m b
#PJM -m e

# run commands
python train.py -c config.json