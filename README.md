This project aims to create an interactive environment for computer vision 
using IPython widgets. It is straightforward to use `ipywidgets` inside 
a Jupyter notebook to control the output of a single function using 
[`interact` or `interactive`](http://ipywidgets.readthedocs.io/en/latest/examples/Using%20Interact.html).
There is no standard way, however, for incorporating widgets with a series of operations.

The project provides a way of representing operations in terms of a directed acyclic graph (DAG). 
We can assign widgets to the input parameters of each operation, and the widgets are linked together
to allow the change in a parameter of one operation to be propagated to other dependent operations.

The operations graph is specified as a nested bracketed string based on the `nltk` package.
For example: 
```
"""
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
"""
```

To get started look at `samples/detect_blobs.py` (noninteractive) and 
`samples/detect_blobs.ipynb` (interactive) for a demo application.
