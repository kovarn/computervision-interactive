"""
Provides interactive versions of different OpenCV operations.
Parameters can be controlled with IPython widgets.

The operations to apply are specified as a DAG. For simplicity,
and to avoid writing implementation for DAG parsing, we use the
ParentedTree class from nltk library with a little hack to process
DAGs instead of trees.

Look at samples/detect_blobs.py and samples/detect_blobs.ipynb for
example non-interactive and interactive usages respectively.
"""

import math
import re
from collections import defaultdict, Counter, deque
from functools import partial
from io import BytesIO
from typing import Union, DefaultDict, Dict, Tuple, Callable

import cv2
import ipywidgets as widgets
import numpy as np
import traitlets
from IPython.core.display import display
from PIL import Image
from nltk import ParentedTree

import filters

ImageArray = np.ndarray
GrayScaleImage = np.ndarray
BinaryImage = np.ndarray


class BaseOperations:
    @staticmethod
    def get_input(image_tuple, index=0):
        return image_tuple[index]

    @classmethod
    def roi(cls, src_widget, horizontal, vertical, zoom, update=True):
        src = cls.get_input(src_widget)
        height = src.shape[0]
        width = src.shape[1]
        center_horizontal = width * horizontal / 100
        center_vertical = height * vertical / 100
        import math
        left = center_horizontal - width / 2 / math.sqrt(zoom)
        right = center_horizontal + width / 2 / math.sqrt(zoom)
        top = center_vertical - height / 2 / math.sqrt(zoom)
        bottom = center_vertical + height / 2 / math.sqrt(zoom)
        if len(src.shape) == 3:
            dst = src[max(0, round(top)):round(bottom), max(0, round(left)): round(right), :]
        else:
            dst = src[max(0, round(top)):round(bottom), max(0, round(left)): round(right)]
        return dst,

    @classmethod
    def gaussianblur(cls, src_widget, ksize, sigmaX=0., sigmaY=0., update=True):
        src = cls.get_input(src_widget)
        dst = cv2.GaussianBlur(src=src, ksize=(3, ksize), sigmaX=sigmaX, sigmaY=sigmaY)
        return dst,

    @classmethod
    def kuwahara_filter(cls, src_widget, ksize, update=True):
        src = cls.get_input(src_widget)
        if len(src.shape) == 3:
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        dst = filters.kuwahara_filter(src=src, kernel_size=ksize)
        return dst,

    @classmethod
    def canny(cls, src_widget, threshold1, threshold2, update=True):
        src = cls.get_input(src_widget)
        if len(src.shape) == 3:
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        dst = cv2.Canny(image=src, threshold1=threshold1, threshold2=threshold2)
        return dst,

    @classmethod
    def houghlines(cls, src_widget, hough_threshold, min_line_length, max_line_gap, resolution_rho, resolution_theta,
                   update=True):
        src = cls.get_input(src_widget)
        # If image is not grayscale convert to grayscale
        if len(src.shape) == 3:
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        lines = cv2.HoughLinesP(image=src, rho=resolution_rho, theta=resolution_theta, threshold=hough_threshold,
                                minLineLength=min_line_length, maxLineGap=max_line_gap)
        a, b, c = lines.shape
        cdst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        for i in range(a):
            cv2.line(cdst, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3,
                     cv2.LINE_AA)
        return cdst,

    @classmethod
    def erode(cls, src_widget, kernel_size, iterations, update=True):
        src = cls.get_input(src_widget)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dst = cv2.erode(src=src, kernel=kernel, iterations=iterations)
        return dst,

    @classmethod
    def dilate(cls, src_widget, kernel_size, iterations, update=True):
        src = cls.get_input(src_widget)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dst = cv2.dilate(src=src, kernel=kernel, iterations=iterations)
        return dst,

    @classmethod
    def closing(cls, src_widget, kernel_size, update=True):
        src = cls.get_input(src_widget)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dst = cv2.morphologyEx(src=src, op=cv2.MORPH_CLOSE, kernel=kernel)

        return dst,

    @classmethod
    def opening(cls, src_widget, kernel_size, update=True):
        src = cls.get_input(src_widget)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dst = cv2.morphologyEx(src=src, op=cv2.MORPH_OPEN, kernel=kernel)
        return dst,

    @classmethod
    def dilate_asym(cls, src_widget, kernel_width, kernel_height, iterations, update=True):
        src = cls.get_input(src_widget)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
        dst = cv2.dilate(src=src, kernel=kernel, iterations=iterations)
        return dst,

    @classmethod
    def erode_asym(cls, src_widget, kernel_width, kernel_height, iterations, update=True):
        src = cls.get_input(src_widget)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
        dst = cv2.erode(src=src, kernel=kernel, iterations=iterations)
        return dst,

    @classmethod
    def blend2(cls, src_widgets, alpha, update=True):
        src1, src2 = (cls.get_input(src_widget) for src_widget in src_widgets)
        if len(src1.shape) == 2 and len(src2.shape) == 3:
            src1 = cv2.cvtColor(src1, cv2.COLOR_GRAY2BGR)
        if len(src2.shape) == 2 and len(src1.shape) == 3:
            src2 = cv2.cvtColor(src2, cv2.COLOR_GRAY2BGR)
        dst = cv2.addWeighted(src1=src1, alpha=alpha, src2=src2, beta=1 - alpha, gamma=0)
        return dst,

    @classmethod
    def draw_fg_on_bg(cls, src_widgets, update=True):
        fg, bg = (cls.get_input(src_widget) for src_widget in src_widgets)
        iscolored_fg = (len(fg.shape) == 3)
        iscolored_bg = (len(bg.shape) == 3)
        if not iscolored_bg:
            bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
            iscolored_bg = True
        if iscolored_fg:
            mask = cv2.bitwise_not(cv2.bitwise_or(cv2.bitwise_or(fg[:, :, 0], fg[:, :, 1]), fg[:, :, 2]))
        else:
            mask = cv2.bitwise_not(fg)
        bg_masked = cv2.bitwise_and(bg, bg, mask=mask)
        if iscolored_bg and not iscolored_fg:
            fg_color = 2
            fg_colored = np.zeros_like(bg)
            fg_colored[:, :, fg_color] = fg
            fg = fg_colored
        dst = cv2.add(fg, bg_masked)
        return dst,

    @classmethod
    def add2onbg(cls, src_widgets, update=True):
        src1, src2, bg = (cls.get_input(src_widget) for src_widget in src_widgets)
        fg = cv2.add(src1, src2)
        return BaseOperations.draw_fg_on_bg(((fg, 2), (bg,)))

    @classmethod
    def threshold(cls, src_widget, threshold, threshold_type, update=True):
        src = cls.get_input(src_widget)
        threshold_types = {'binary': cv2.THRESH_BINARY, 'binary_inv': cv2.THRESH_BINARY_INV}
        thresh, dst = cv2.threshold(src, threshold, 255, threshold_type)
        return dst,

    @classmethod
    def adaptive_threshold(cls, src_widget, method, threshold_type, window_size, threshold, update=True):
        src = cls.get_input(src_widget)
        dst = cv2.adaptiveThreshold(src, 255, method, threshold_type, window_size, threshold)
        return dst,

    @classmethod
    def components(cls, src_widget, update=True):
        src = cls.get_input(src_widget)
        connectivity = 4
        outputs = cv2.connectedComponents(src, connectivity=connectivity, ltype=cv2.CV_32S)
        # print(outputs[0])
        return outputs[1], outputs[0]

    @classmethod
    def components_with_stats(cls, src_widget, update=True):
        src = cls.get_input(src_widget)
        connectivity = 4
        output = cv2.connectedComponentsWithStats(src, connectivity=connectivity, ltype=cv2.CV_32S)
        # print(output[0])
        # Get the results
        # The first cell is the number of labels
        num_labels = output[0]
        # The second cell is the label matrix
        labels = output[1]
        # The third cell is the stat matrix
        stats = output[2]
        # The fourth cell is the centroid matrix
        centroids = output[3]
        dst = np.zeros_like(src, dtype=np.uint8)
        boxes = stats[1:, :4]
        boxes[:, 2:4] += boxes[:, :2]
        for x in boxes:
            cv2.rectangle(dst, tuple(x[:2]), tuple(x[2:]), 255, 3)
        # dst = image_type.draw_fg_on_bg(((dst,),(cv2.cvtColor(src,cv2.COLOR_GRAY2BGR),)))[0]
        return dst, num_labels


class WidgetOperations(BaseOperations):
    @staticmethod
    def get_input(image_widget, index=0):
        return image_widget.result[index]


ImageWidget = widgets.Widget
ImageTuple = Tuple[ImageArray]
OpWidget = Union[ImageWidget, ImageTuple]


class ApplyOperations:
    def __init__(self, interactive_mode: bool):
        self.interactive_mode = interactive_mode
        if interactive_mode:
            self.interactive = partial(
                widgets.interactive, update=True, __output_result=False)  # type: Callable[[...], OpWidget]
            self.IntSlider = partial(widgets.IntSlider, continuous_update=False)
            self.FloatSlider = partial(widgets.FloatSlider, continuous_update=False)
            self.IntRangeSlider = partial(widgets.IntRangeSlider, continuous_update=False)
            self.Dropdown = widgets.Dropdown
            self.fixed = widgets.fixed
            self.operations = WidgetOperations

        else:
            self.interactive = call_with_kwargs  # type: Callable[[...], OpWidget]
            self.IntSlider = get_value
            self.FloatSlider = get_value
            self.IntRangeSlider = get_value
            self.Dropdown = get_value
            self.fixed = lambda x: x
            self.operations = BaseOperations

    def applyop(self, op: str, image_widget, *args) -> Union['ImageControls', OpWidget]:
        print_result = False

        if op == 'roi':
            defaults = set_defaults([50, 50, 1], args)
            widget = self.interactive(self.operations.roi, src_widget=self.fixed(image_widget),
                                      horizontal=self.IntSlider(defaults[0], 0, 100, 1),
                                      vertical=self.IntSlider(defaults[1], 0, 100, 1),
                                      zoom=self.IntSlider(defaults[2], 1, 16, 1))
        elif op == 'gaussianblur':
            defaults = set_defaults([3], args)
            widget = self.interactive(self.operations.gaussianblur, src_widget=self.fixed(image_widget),
                                      ksize=self.IntSlider(defaults[0], 1, 15, 2), sigmaX=self.fixed(0.),
                                      sigmaY=self.fixed(0.))
        elif op == 'kuwahara':
            defaults = set_defaults([5], args)
            widget = self.interactive(self.operations.kuwahara_filter, src_widget=self.fixed(image_widget),
                                      ksize=self.IntSlider(defaults[0], 1, 15, 2))
        elif op == 'canny':
            defaults = set_defaults([100, 200], args)
            widget = self.interactive(self.operations.canny, src_widget=self.fixed(image_widget),
                                      threshold1=self.IntSlider(defaults[0], 0, 255, 1),
                                      threshold2=self.IntSlider(defaults[1], 0, 255, 1))
        elif op == 'houghlines':
            defaults = set_defaults([40, 50, 10, 3, math.pi / 180.0], args)
            widget = self.interactive(self.operations.houghlines, src_widget=self.fixed(image_widget),
                                      hough_threshold=self.IntSlider(defaults[0], 1, 255, 1),
                                      min_line_length=self.IntSlider(defaults[1], 1, 255, 1),
                                      max_line_gap=self.IntSlider(defaults[2], 1, 255, 1),
                                      resolution_rho=self.fixed(defaults[3]),
                                      resolution_theta=self.fixed(defaults[4]))
        elif op == 'opening':
            defaults = set_defaults([5], args)
            widget = self.interactive(self.operations.opening, src_widget=self.fixed(image_widget),
                                      kernel_size=self.IntSlider(defaults[0], 1, 21, 1))
        elif op == 'closing':
            defaults = set_defaults([5], args)
            widget = self.interactive(self.operations.closing, src_widget=self.fixed(image_widget),
                                      kernel_size=self.IntSlider(defaults[0], 1, 21, 1))
        elif op == 'dilate':
            defaults = set_defaults([5, 1], args)
            widget = self.interactive(self.operations.dilate, src_widget=self.fixed(image_widget),
                                      kernel_size=self.IntSlider(defaults[0], 1, 21, 1),
                                      iterations=self.fixed(defaults[1]))
        elif op == 'erode':
            defaults = set_defaults([5, 1], args)
            widget = self.interactive(self.operations.erode, src_widget=self.fixed(image_widget),
                                      kernel_size=self.IntSlider(defaults[0], 1, 21, 1),
                                      iterations=self.fixed(defaults[1]))
        elif op == 'dilate_asym':
            defaults = set_defaults([11, 1, 1], args)
            widget = self.interactive(self.operations.dilate_asym, src_widget=self.fixed(image_widget),
                                      kernel_width=self.IntSlider(defaults[0], 1, 21, 1),
                                      kernel_height=self.IntSlider(defaults[1], 1, 21, 1),
                                      iterations=self.fixed(defaults[2]))
        elif op == 'erode_asym':
            defaults = set_defaults([5, 1, 1], args)
            widget = self.interactive(self.operations.erode_asym, src_widget=self.fixed(image_widget),
                                      kernel_width=self.IntSlider(defaults[0], 1, 21, 1),
                                      kernel_height=self.IntSlider(defaults[1], 1, 21, 1),
                                      iterations=self.fixed(defaults[2]))
        elif op == 'blend2':
            defaults = set_defaults([0.5], args)
            widget = self.interactive(self.operations.blend2, src_widgets=self.fixed(image_widget),
                                      alpha=self.FloatSlider(value=defaults[0], min=0, max=1))
        elif op == 'fgonbg':
            widget = self.interactive(self.operations.draw_fg_on_bg, src_widgets=self.fixed(image_widget))
        elif op == 'add2onbg':
            widget = self.interactive(self.operations.add2onbg, src_widgets=self.fixed(image_widget))
        elif op == 'threshold':
            defaults = set_defaults([127, 'binary'], args)
            threshold_type_options = {'binary': cv2.THRESH_BINARY,
                                      'binary_inv': cv2.THRESH_BINARY_INV}
            widget = self.interactive(self.operations.threshold, src_widget=self.fixed(image_widget),
                                      threshold=self.IntSlider(defaults[0], 0, 255, 1),
                                      threshold_type=self.Dropdown(value=threshold_type_options[defaults[1]],
                                                                   options=threshold_type_options))
        elif op == 'adaptive_threshold':
            defaults = set_defaults(['mean', 'binary_inv', 11, 2], args)
            method_options = {'mean': cv2.ADAPTIVE_THRESH_MEAN_C,
                              'gaussian': cv2.ADAPTIVE_THRESH_GAUSSIAN_C}
            threshold_type_options = {'binary': cv2.THRESH_BINARY,
                                      'binary_inv': cv2.THRESH_BINARY_INV}
            widget = self.interactive(self.operations.adaptive_threshold, src_widget=self.fixed(image_widget),
                                      method=self.Dropdown(value=method_options[defaults[0]],
                                                           options=method_options),
                                      threshold_type=self.Dropdown(value=threshold_type_options[defaults[1]],
                                                                   options=threshold_type_options),
                                      window_size=self.IntSlider(defaults[2], 1, 41, 2),
                                      threshold=self.IntSlider(defaults[3], 0, 255, 1))

        elif op == 'components':  # defaults = set_defaults([127], args)
            widget = self.interactive(self.operations.components, src_widget=self.fixed(image_widget))
            print_result = True
        elif op == 'components_with_stats':  # defaults = set_defaults([127], args)
            widget = self.interactive(self.operations.components_with_stats,
                                      src_widget=self.fixed(image_widget))
            print_result = True
        else:
            raise ValueError('Unknown operation: ' + op)

        if self.interactive_mode:
            control = ImageControls(widget, print_result)
            control.name = op

            control.display()
            control.update_image_widget()
            return control
        else:
            return widget

    def applyops(self, ops_tree: ParentedTree, src: ImageArray):
        # print('yay')
        node_widgets = dict()
        node_input_widgets_array = defaultdict(
            dict)  # type: DefaultDict[str, Dict[int, Union[ImageControls, ImageTuple]]

        for (ix, node) in enumerate(traverse(ops_tree)):
            op = OpWithParams(node)
            if op.skip:
                if op.has_siblings:
                    raise NotImplementedError('Cannot skip node with siblings.')
                node_widgets[op.node_id] = node_widgets[op.parent_id]
                continue
            # node_name = str(ix) + op
            if op.has_siblings:
                # this node will have multiple inputs
                node_input_widgets_array[op.sibling_group_id][op.sibling_order] = node_widgets[op.parent_id]
                if op.is_last_sibling:
                    # we have collected all parents
                    # our traversing function ensures this
                    node_parents_widgets = node_input_widgets_array[
                        op.sibling_group_id]  # type: Dict[int, Union[ImageControls, ImageTuple]]
                    if self.interactive_mode:
                        node_src_widgets = [node_parents_widgets[i].control_widget for i in
                                            sorted(node_parents_widgets)]
                    else:
                        node_src_widgets = [node_parents_widgets[i] for i in sorted(node_parents_widgets)]
                    widget = self.applyop(op.op_name, node_src_widgets, *op.params)
                    if self.interactive_mode:
                        for w in node_src_widgets:
                            link_widgets(w, widget.control_widget)
                    node_widgets[op.node_id] = widget
                    yield widget
            else:
                if self.interactive_mode:
                    node_src_widget = dummy(src) if op.parent_id is None else node_widgets[op.parent_id].control_widget
                else:
                    node_src_widget = (src,) if op.parent_id is None else node_widgets[op.parent_id]
                widget = self.applyop(op.op_name, node_src_widget, *op.params)
                if self.interactive_mode:
                    if op.parent_id is not None:
                        link_widgets(node_src_widget, widget.control_widget)
                # cv2.imwrite(newfolder + '/' + node_name + '.png', dst)
                node_widgets[op.node_id] = widget
                # plt.figure()
                yield widget


def call_with_kwargs(f, **kwargs):
    return f(**kwargs)


def get_value(value, *args, **kwargs):
    if '__map' in kwargs:
        return kwargs['__map'](value)
    else:
        return value


def set_defaults(defaults, args):
    defaults[:len(args)] = (type(d)(a) for d, a in zip(defaults, args))
    return defaults


class ImageControls:
    def __init__(self, control_widget, print_result=False):
        self.print_result = print_result
        self.control_widget = control_widget
        self.image_widget = widgets.Image(format='png', width='70%')
        self.print_widget = widgets.HTML()
        self.name = ''
        self.update_image_widget()
        for w in self.control_widget.children:
            w.observe(self.update_image_widget, 'value')

    def update_image_widget(self, *args):
        self.print_widget.value = '<h4>' + self.name.title() + '</h4>'
        if self.control_widget.result is None:
            img = np.zeros([4, 4, 3], dtype=np.uint8)
        else:
            img = self.control_widget.result[0]
            if self.print_result and len(self.control_widget.result) > 1:
                self.print_widget.value += str(self.control_widget.result[1])
        self.image_widget.value = array_to_img(img)

    def display(self):
        self.print_widget.layout.margin = '0px 0px 0px 40px'
        vbox = widgets.VBox([self.print_widget, self.control_widget])
        vbox.layout.margin = '0px 0px 0px 40px'
        display(widgets.HBox([self.image_widget, vbox]))


def array_to_img(img: ImageArray):
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = img[:, :, ::-1]
    im = Image.fromarray(img)
    bio = BytesIO()
    im.save(bio, format='png')
    return bio.getvalue()


def link_widgets(src_widget, dst_widget):
    def toggle_dst_value(src_value):
        return not dst_widget.children[-1].value

    traitlets.dlink((src_widget.children[-1], "value"), (dst_widget.children[-1], "value"), toggle_dst_value)


class dummy:
    """
    dummy widget class
    """

    def __init__(self, image):
        self.result = (image,)


def traverse(dag: ParentedTree, children=iter):
    node_parents = Counter()
    queue = deque([dag])
    while queue:
        node = queue.popleft()  # type: ParentedTree
        label = node.label()  # type: str
        label_groups = re.findall(r"([^\s,]+)[\s,]*", label)
        prefix_numbers = re.findall(r"([^-]+)", label_groups[0])
        op = prefix_numbers[-1]
        if len(prefix_numbers) > 1:
            # this node will have multiple inputs
            node_id = ''.join(prefix_numbers[1:])
            if prefix_numbers[0] == prefix_numbers[1]:
                if node_parents[node_id] < int(prefix_numbers[1]) - 1:
                    queue.append(node)
                    continue
            else:
                node_parents[node_id] += 1
        yield node
        try:
            queue.extend(children(node))
        except TypeError:
            pass


class OpWithParams:
    def __init__(self, node):
        self.node_id = node.treeposition()
        self.parent_id = None if node.parent() is None else node.parent().treeposition()
        label = node.label()
        self.skip = (label[0] == '#')
        label_groups = re.findall(r"([^\s,]+)[\s,]*", label)
        self.params = label_groups[1:]
        prefix_numbers = re.findall(r"([^-]+)", label_groups[0])
        self.op_name = prefix_numbers.pop()  # last item is op name, rest will be numbers
        if len(prefix_numbers) > 0:
            # this node will have multiple inputs
            self.has_siblings = True
            self.sibling_group_id = ''.join(prefix_numbers[1:]) + self.op_name  # common among all siblings
            self.sibling_order = int(prefix_numbers[0])
            self.is_last_sibling = (prefix_numbers[0] == prefix_numbers[1])
        else:
            self.has_siblings = False
