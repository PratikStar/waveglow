#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=share
#PJM -L gpu=4
#PJM -N waveglow
#PJM -j
#PJM -m b
#PJM -m e

# run commands
python distributed.py -c config.json