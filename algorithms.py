from pathlib import Path
from skimage import exposure
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import skimage
import ipywidgets as widgets
from IPython.display import display
from skimage.segmentation import clear_border, watershed
from skimage.morphology import white_tophat, disk, remove_small_objects
from skimage.filters import threshold_otsu
from skimage.segmentation import active_contour
from skimage.measure import find_contours, label, regionprops
from skimage.draw import polygon
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_edt

matplotlib.rcParams['font.size'] = 8

def chromosome_mask_with_active_contour(img_path,
                                         gaussian_sigma=0.3,
                                         clahe_clip_limit=0.03,
                                         tophat_disk_size=15,
                                         snake_sigma=3,
                                         alpha=0.015,
                                         beta=10,
                                         w_line=0,
                                         w_edge=1,
                                         gamma=0.001,
                                         max_num_iter=10,
                                         min_object_size=600,
                                         plot=True):
    """
    Segment chromosomes using scikit-image active_contour (snake).
    An initial mask from CLAHE + top-hat thresholding seeds one snake per
    chromosome; each snake is then refined against image edges.

    Parameters
    ----------
    img_path          : str, Path, or array
    gaussian_sigma    : float — pre-smoothing (default 0.3)
    clahe_clip_limit  : float — CLAHE contrast limit (default 0.03)
    tophat_disk_size  : int   — top-hat SE radius (default 15)
    snake_sigma       : float — Gaussian blur applied before snake gradient (default 3)
    alpha             : float — snake tension / length penalty (default 0.015)
    beta              : float — snake rigidity / curvature penalty (default 10)
    w_line            : float — attraction to bright regions (default 0)
    w_edge            : float — attraction to edges (default 1)
    gamma             : float — time-step size (default 0.001)
    max_num_iter      : int   — max snake iterations per chromosome (default 2500)
    min_object_size   : int   — min blob area to keep (default 600)
    plot              : bool

    Returns
    -------
    mask : np.ndarray (bool)
    """

    if isinstance(img_path, (str, Path)):
        img_raw = skimage.io.imread(img_path, as_gray=True)
    else:
        img_raw = img_path.copy()


    img = skimage.filters.gaussian(img_raw, gaussian_sigma, preserve_range=True)
    img = np.clip(img, 0, 1)

    img_adapteq = exposure.equalize_adapthist(img, clip_limit=clahe_clip_limit)
    inverted    = 1.0 - img_adapteq
    tophat      = white_tophat(inverted, disk(tophat_disk_size))

    thresh     = threshold_otsu(tophat)
    seed_mask  = tophat > thresh
    seed_mask  = remove_small_objects(seed_mask, min_size=min_object_size)
    seed_mask  = clear_border(seed_mask)

    img_snake = skimage.filters.gaussian(inverted, snake_sigma)

    labeled    = label(seed_mask)
    final_mask = np.zeros(img.shape, dtype=bool)

    for region in regionprops(labeled):
        region_bin = (labeled == region.label).astype(float)
        contours   = find_contours(region_bin, 0.5)
        if not contours:
            continue

        snake_init = contours[np.argmax([len(c) for c in contours])]

        try:
            snake = active_contour(img_snake, snake_init,
                                   alpha=alpha, beta=beta,
                                   w_line=w_line, w_edge=w_edge,
                                   gamma=gamma, max_num_iter=max_num_iter)

          
            snake[:, 0] = np.clip(snake[:, 0], 0, img.shape[0] - 1)
            snake[:, 1] = np.clip(snake[:, 1], 0, img.shape[1] - 1)
            rr, cc      = polygon(snake[:, 0], snake[:, 1], img.shape)
            final_mask[rr, cc] = True

        except Exception:
            final_mask[region.coords[:, 0], region.coords[:, 1]] = True

    final_mask = remove_small_objects(final_mask, min_size=min_object_size)
    final_mask = clear_border(final_mask)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title("Preprocessed")
        axes[1].imshow(final_mask, cmap='gray')
        axes[1].set_title("Binary mask (active contour)")
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    return final_mask

def chromosome_mask_with_contrast_stretching(img_path,
                            gaussian_sigma=0.3,
                            tophat_disk_size=15,
                            block_size=101,
                            opening_disk_size=5,
                            min_object_size=600,
                            plot=True,with_otsu=False):
    """
    Segment chromosomes from a grayscale microscopy image.

    Parameters
    ----------
    img_path          : str, Path, or array — file path or preloaded image array
    gaussian_sigma    : float — smoothing strength (default 0.77)
    tophat_disk_size  : int   — structuring element radius for top-hat (default 30)
    block_size        : int   — local threshold window, must be odd (default 73)
    opening_disk_size : int   — morphological opening radius (default 30)
    min_object_size   : int   — minimum blob area in pixels to keep (default 500)
    plot              : bool  — show side-by-side figure (default True)

    Returns
    -------
    mask : np.ndarray (bool) — binary chromosome mask
    """

    if isinstance(img_path, (str, Path)):
        img_raw = skimage.io.imread(img_path, as_gray=True)
    else:
        img_raw = img_path.copy()

    img = skimage.filters.gaussian(img_raw, gaussian_sigma, preserve_range=True)
    img = np.clip(img, 0, 1)

   
    img_rescale = exposure.rescale_intensity(img)

   
    inverted = 1.0 - img_rescale

   
    tophat = white_tophat(inverted, disk(tophat_disk_size))

    if with_otsu:

        local_thresholds =  threshold_otsu(tophat)

    else:

        local_thresholds = skimage.filters.threshold_local(tophat, block_size, method='gaussian')

    
    binary_opened = skimage.morphology.opening(inverted, disk(opening_disk_size))
    
    mask = binary_opened > local_thresholds

  
    mask = remove_small_objects(mask, min_size=min_object_size)
    mask = clear_border(mask)

   
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title("Preprocessed")
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Binary mask")
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    return mask

def chromosome_mask_with_adaptive_histogram(img_path, 
                        gaussian_sigma=0.3,
                        clahe_clip_limit=0.03,
                        tophat_disk_size=15,
                        block_size=101,
                        opening_disk_size=5,
                        min_object_size=600,
                        min_distance=10,
                        plot=True,with_otsu=True):
    """
    Segment chromosomes from a grayscale microscopy image.

    Parameters
    ----------
    img_path        : str or array — file path or preloaded image array
    gaussian_sigma  : float — smoothing strength (default 0.3)
    clahe_clip_limit: float — CLAHE contrast limit (default 0.03)
    tophat_disk_size: int   — structuring element radius for top-hat (default 15)
    block_size      : int   — local threshold window, must be odd (default 101)
    opening_disk_size: int  — morphological opening radius (default 5)
    min_object_size : int   — minimum blob area in pixels to keep (default 600)
    plot            : bool  — show side-by-side figure (default True)

    Returns
    -------
    mask : np.ndarray (bool) — binary chromosome mask
    """

    if isinstance(img_path, (str, Path)):
        img_raw = skimage.io.imread(img_path, as_gray=True)
    else:
        img_raw = img_path.copy()

   
    img = skimage.filters.gaussian(img_raw, gaussian_sigma, preserve_range=True)
    img = np.clip(img, 0, 1)


    img_adapteq = exposure.equalize_adapthist(img, clip_limit=clahe_clip_limit)


    inverted = 1.0 - img_adapteq

    
    tophat = white_tophat(inverted, disk(tophat_disk_size))

    if with_otsu:

        local_thresholds =  threshold_otsu(tophat)

    else:

        local_thresholds = skimage.filters.threshold_local(tophat, block_size, method='gaussian')


    binary_local = tophat > local_thresholds


    binary_opened = skimage.morphology.opening(binary_local, disk(opening_disk_size))
    binary_opened = remove_small_objects(binary_opened, min_size=min_object_size)
    binary_opened = clear_border(binary_opened)

    # Watershed to split clustered chromosomes
    distance = distance_transform_edt(binary_opened)
    coords = peak_local_max(distance, min_distance=min_distance, labels=binary_opened)
    peak_mask = np.zeros(distance.shape, dtype=bool)
    peak_mask[tuple(coords.T)] = True
    markers = label(peak_mask)
    labels_ws = watershed(-distance, markers, mask=binary_opened)
    mask = labels_ws > 0

    if plot:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title("Preprocessed")
        axes[1].imshow(binary_opened, cmap='gray')
        axes[1].set_title("Before watershed")
        axes[2].imshow(distance, cmap='hot')
        axes[2].plot(coords[:, 1], coords[:, 0], 'c.', markersize=4)
        axes[2].set_title(f"Distance transform + peaks (min_dist={min_distance})")
        axes[3].imshow(labels_ws, cmap='nipy_spectral')
        axes[3].set_title(f"After watershed ({labels_ws.max()} objects)")
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    return mask


class _WatershedEditor:
    """
    Interactive matplotlib editor for correcting watershed segmentation.

    Usage in Jupyter (after %matplotlib widget):
        editor = _WatershedEditor(img, binary_mask, min_distance=10)
        # click to place seeds, then in a new cell:
        mask = editor.mask

    Controls
    --------
    Left-click  : add a seed to split a merged blob
    Right-click : remove the nearest user seed
    """

    def __init__(self, img, binary_mask, min_distance):
        self.img = img
        self.binary_mask = binary_mask
        self.min_distance = min_distance
        self.user_seeds = []

        self._run_watershed()

        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 6))
        self.fig.canvas.toolbar_visible = False
        self._draw()
        plt.tight_layout()

        self._btn = widgets.Button(description='Done ✓',
                                   button_style='success',
                                   layout=widgets.Layout(width='120px'))
        self._out = widgets.Output()
        self._btn.on_click(self._on_done)
        display(widgets.VBox([self.fig.canvas, self._btn, self._out]))

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

    def _run_watershed(self):
        self.distance = distance_transform_edt(self.binary_mask)
        coords = peak_local_max(self.distance, min_distance=self.min_distance,
                                labels=self.binary_mask)
        peak_mask = np.zeros(self.distance.shape, dtype=bool)
        if coords.size:
            peak_mask[tuple(coords.T)] = True
        for r, c in self.user_seeds:
            peak_mask[r, c] = True
        markers = label(peak_mask)
        self.labels_ws = watershed(-self.distance, markers, mask=self.binary_mask)
        self.mask = self.labels_ws > 0

    def _draw(self):
        for ax in self.axes:
            ax.clear()
        self.axes[0].imshow(self.img, cmap='gray')
        self.axes[0].imshow(self.mask, cmap='Reds', alpha=0.35)
        if self.user_seeds:
            ys, xs = zip(*self.user_seeds)
            self.axes[0].plot(xs, ys, 'b+', markersize=12, markeredgewidth=2)
        self.axes[0].set_title("Left-click: add seed | Right-click: remove seed")
        self.axes[1].imshow(self.labels_ws, cmap='nipy_spectral')
        self.axes[1].set_title(f"Watershed  —  {self.labels_ws.max()} objects")
        for ax in self.axes:
            ax.axis('off')
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        print(f"click: button={event.button} inaxes={event.inaxes is not None} x={event.xdata}")
        if event.inaxes != self.axes[0] or event.xdata is None:
            return
        r = int(np.clip(round(event.ydata), 0, self.binary_mask.shape[0] - 1))
        c = int(np.clip(round(event.xdata), 0, self.binary_mask.shape[1] - 1))
        if event.button == 1:
            self.user_seeds.append((r, c))
        elif event.button == 3 and self.user_seeds:
            dists = [(r - sr) ** 2 + (c - sc) ** 2 for sr, sc in self.user_seeds]
            self.user_seeds.pop(int(np.argmin(dists)))
        self._run_watershed()
        self._draw()

    def _on_done(self, _):
        self._btn.disabled = True
        self._btn.description = 'Saved ✓'
        plt.close(self.fig)
        with self._out:
            _, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(self.labels_ws, cmap='nipy_spectral')
            ax.set_title(f'Final mask — {self.labels_ws.max()} objects')
            ax.axis('off')
            plt.tight_layout()
            plt.show()


def interactive_watershed_correction(img_path,
                                     gaussian_sigma=0.3,
                                     clahe_clip_limit=0.03,
                                     tophat_disk_size=15,
                                     block_size=101,
                                     opening_disk_size=5,
                                     min_object_size=600,
                                     min_distance=10,
                                     with_otsu=True):
    """
    Semi-automated chromosome segmentation with interactive seed correction.

    Returns a _WatershedEditor object immediately. Interact with the figure
    (left-click to add seeds, right-click to remove), then access .mask
    in a new cell when done.

    Example
    -------
    editor = interactive_watershed_correction(img_path)
    # ... click to fix merged blobs ...
    mask = editor.mask   # run in a new cell when satisfied

    Controls
    --------
    Left-click  : add a watershed seed on a merged cluster to split it
    Right-click : remove the nearest user-placed seed

    Parameters
    ----------
    img_path         : str, Path, or array
    gaussian_sigma   : float  (default 0.3)
    clahe_clip_limit : float  (default 0.03)
    tophat_disk_size : int    (default 15)
    block_size       : int    (default 101, must be odd)
    opening_disk_size: int    (default 5)
    min_object_size  : int    (default 600)
    min_distance     : int    min pixels between auto-detected peaks (default 10)
    with_otsu        : bool   (default True)

    Returns
    -------
    editor : _WatershedEditor  — access editor.mask when done
    """
    if isinstance(img_path, (str, Path)):
        img_raw = skimage.io.imread(img_path, as_gray=True)
    else:
        img_raw = np.array(img_path, dtype=float)

    img = skimage.filters.gaussian(img_raw, gaussian_sigma, preserve_range=True)
    img = np.clip(img, 0, 1)

    img_adapteq = exposure.equalize_adapthist(img, clip_limit=clahe_clip_limit)
    inverted    = 1.0 - img_adapteq
    tophat      = white_tophat(inverted, disk(tophat_disk_size))

    if with_otsu:
        thresh = threshold_otsu(tophat)
    else:
        thresh = skimage.filters.threshold_local(tophat, block_size, method='gaussian')

    binary_local  = tophat > thresh
    binary_opened = skimage.morphology.opening(binary_local, disk(opening_disk_size))
    binary_opened = remove_small_objects(binary_opened, min_size=min_object_size)
    binary_opened = clear_border(binary_opened)

    return _WatershedEditor(img, binary_opened, min_distance)


def napari_watershed_correction(img_path,
                                gaussian_sigma=0.3,
                                clahe_clip_limit=0.03,
                                tophat_disk_size=15,
                                block_size=101,
                                opening_disk_size=5,
                                min_object_size=600,
                                min_distance=10,
                                with_otsu=True):
    """
    Semi-automated chromosome segmentation using napari.

    Runs the automated CLAHE + top-hat + watershed pipeline, then opens
    napari with the image and segmentation labels so you can manually
    correct merged chromosomes using the paintbrush tool.

    Requires: pip install napari[all]

    Usage
    -----
    viewer, labels_layer = napari_watershed_correction(img_path)
    # Edit the 'segmentation' layer in napari using the paintbrush
    # Then get the corrected mask:
    mask = labels_layer.data > 0

    Parameters
    ----------
    img_path         : str, Path, or array
    gaussian_sigma   : float  (default 0.3)
    clahe_clip_limit : float  (default 0.03)
    tophat_disk_size : int    (default 15)
    block_size       : int    (default 101, must be odd)
    opening_disk_size: int    (default 5)
    min_object_size  : int    (default 600)
    min_distance     : int    (default 10)
    with_otsu        : bool   (default True)

    Returns
    -------
    viewer       : napari.Viewer
    labels_layer : napari labels layer  — edit this, then access .data
    """
    import napari

    if isinstance(img_path, (str, Path)):
        img_raw = skimage.io.imread(img_path, as_gray=True)
    else:
        img_raw = np.array(img_path, dtype=float)

    img = skimage.filters.gaussian(img_raw, gaussian_sigma, preserve_range=True)
    img = np.clip(img, 0, 1)

    img_adapteq = exposure.equalize_adapthist(img, clip_limit=clahe_clip_limit)
    inverted    = 1.0 - img_adapteq
    tophat      = white_tophat(inverted, disk(tophat_disk_size))

    if with_otsu:
        thresh = threshold_otsu(tophat)
    else:
        thresh = skimage.filters.threshold_local(tophat, block_size, method='gaussian')

    binary_local  = tophat > thresh
    binary_opened = skimage.morphology.opening(binary_local, disk(opening_disk_size))
    binary_opened = remove_small_objects(binary_opened, min_size=min_object_size)
    binary_opened = clear_border(binary_opened)

    distance  = distance_transform_edt(binary_opened)
    coords    = peak_local_max(distance, min_distance=min_distance, labels=binary_opened)
    peak_mask = np.zeros(distance.shape, dtype=bool)
    if coords.size:
        peak_mask[tuple(coords.T)] = True
    markers   = label(peak_mask)
    labels_ws = watershed(-distance, markers, mask=binary_opened)

    viewer       = napari.Viewer()
    viewer.add_image(img, name='image', colormap='gray')
    labels_layer = viewer.add_labels(labels_ws, name='segmentation')

    print("napari is open.")
    print("  - Select the 'segmentation' layer")
    print("  - Use the paintbrush to split merged chromosomes (paint label 0 between them)")
    print("  - When done, run:  mask = labels_layer.data > 0")

    return viewer, labels_layer