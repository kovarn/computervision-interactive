"""
Detect blobs from input image.
"""

# ++import sys
# ++sys.path.append('..')

import cv2
from matplotlib import pyplot as plt
from nltk.tree import *

from cv_interactive.operations import ApplyOperations
from cv_interactive.utils import imshow

if __name__ == '__main__':
    # +-
    ##
    image_path = "cells.jpg"

    interactive_mode = False
    # +-interactive_mode = True

    ops_tree = ParentedTree.fromstring("""
        (roi 50,50,1
         (2-2-blend2 0.3)
         (2-2-fgonbg
          (1-2-blend2)
         )
         (kuwahara 3
          (erode 10
           (adaptive_threshold mean binary_inv 19 7
            (erode 7
             (components_with_stats
              (1-2-fgonbg)
             )
            )
           )
          )
         )
        )
        """, node_pattern=r"[^()\n]*")

    ##
    src = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.axis('off')
    # imshow(src)
    # plt.show()

    apply_ops = ApplyOperations(interactive_mode)
    outputs = list(apply_ops.applyops(ops_tree, src))

    num_blobs = outputs[-3].control_widget.result[1] - 1 if interactive_mode else outputs[-3][1] - 1
    print(str(num_blobs) + ' blobs detected.')

    if not interactive_mode:
        imshow(outputs[-1][0])
        plt.show()
