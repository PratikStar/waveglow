#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -N waveglow
#PJM -j
#PJM -m b
#PJM -m e

# run commands
python distributed.py -c config.json