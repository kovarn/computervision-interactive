from matplotlib import pyplot as plt


def imshow(img, **kwargs):
    if len(img.shape) == 2 and 'cmap' not in kwargs:
        return plt.imshow(img, cmap=plt.cm.gray, **kwargs)
    if len(img.shape) == 3 and img.shape[2] == 3:
        return plt.imshow(img[:, :, ::-1], **kwargs)
    return plt.imshow(img, **kwargs)
