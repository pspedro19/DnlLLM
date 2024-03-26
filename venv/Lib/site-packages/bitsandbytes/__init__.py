# Copyright (c) Facebook, Inc. and its affiliates. 
#   
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.
from .optim import adam
from .nn import modules
print('='*30 + 'WARNING: DEPRECATED!' + '='*30)
print('WARNING! This version of bitsandbytes is deprecated. Please switch to `pip install bitsandbytes` and the new repo: https://github.com/TimDettmers/bitsandbytes')
print('='*30 + 'WARNING: DEPRECATED!' + '='*30)
__pdoc__ = {'libBitsNBytes' : False,
            'optim.optimizer.Optimizer8bit': False,
            'optim.optimizer.MockArgs': False
           }
