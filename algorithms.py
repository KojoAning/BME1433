from pathlib import Path
from skimage import exposure
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import skimage
import pandas as pd
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
                                         max_num_iter=2500,
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
                            max_object_size=30000,
                            max_eccentricity=0.97,
                            plot=True,with_otsu=False):
    """
    Segment chromosomes using contrast stretching as an contrast enhancement technique.

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

   
    p2, p98     = np.percentile(img, [2, 98])
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

   
    inverted = 1.0 - img_rescale

   
    tophat = white_tophat(inverted, disk(tophat_disk_size))

    if with_otsu:

        local_thresholds =  threshold_otsu(tophat)

    else:

        local_thresholds = skimage.filters.threshold_local(tophat, block_size, method='gaussian')

    
    binary_local  = tophat > local_thresholds

    labeled_pre = label(binary_local)
    for region in regionprops(labeled_pre):
        if region.area > max_object_size:
            binary_local[labeled_pre == region.label] = False

    binary_opened = skimage.morphology.opening(binary_local, disk(opening_disk_size))

    mask = remove_small_objects(binary_opened, min_size=min_object_size)
    mask = clear_border(mask)

    labeled_mask = label(mask)
    for region in regionprops(labeled_mask):
        if region.area > max_object_size or region.eccentricity > max_eccentricity:
            mask[labeled_mask == region.label] = False

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
                        opening_disk_size=7,
                        min_object_size=400,
                        max_object_size=30000,
                        max_eccentricity=0.97,
                        min_distance=3,
                        plot=True,with_otsu=True):
    """
    Segment chromosomes using adaptive histogram equalization as a contrast enhancement technique.

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

    labeled_pre = label(binary_local)
    for region in regionprops(labeled_pre):
        if region.area > max_object_size:
            binary_local[labeled_pre == region.label] = False


    binary_opened = skimage.morphology.opening(binary_local, disk(opening_disk_size))
    binary_opened = remove_small_objects(binary_opened, min_size=min_object_size)
    binary_opened = clear_border(binary_opened)


    gradient  = skimage.filters.sobel(img_adapteq)
    distance  = distance_transform_edt(binary_opened)
    coords    = peak_local_max(distance, min_distance=min_distance, labels=binary_opened)
    peak_mask = np.zeros(distance.shape, dtype=bool)
    peak_mask[tuple(coords.T)] = True
    markers   = label(peak_mask)
    labels_ws = watershed(gradient, markers, mask=binary_opened)
    mask = labels_ws > 0

    labeled_mask = label(mask)
    for region in regionprops(labeled_mask):
        if region.area > max_object_size or region.eccentricity > max_eccentricity:
            mask[labeled_mask == region.label] = False

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

    return binary_opened

def chromosome_mask_thresholding(img_path,
                        gaussian_sigma=0.3,
                        max_object_size=30000,
                        max_eccentricity=0.97,
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


    thresh_mask = img < 0.8
    mask = remove_small_objects(thresh_mask, min_size=100)
    binary_closed = skimage.morphology.closing(mask, disk(3))



    # Remove large blobs (e.g. nuclei) before closing so they don't get fragmented
    labeled_pre = label(binary_closed)
    for region in regionprops(labeled_pre):
        if region.area > max_object_size:
            binary_closed[labeled_pre == region.label] = False

    mask = clear_border(binary_closed)



    # Remove remaining large objects and scale-bar-like artifacts (high eccentricity)
    labeled_mask = label(mask)
    for region in regionprops(labeled_mask):
        if region.area > max_object_size or region.eccentricity > max_eccentricity:
            mask[labeled_mask == region.label] = False

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        axes[0].imshow(img_raw, cmap='gray')
        axes[0].set_title("Preprocessed")
        axes[1].imshow(thresh_mask, cmap='gray')
        axes[1].set_title("thresholding mask")
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title("Final mask")
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    return mask


def napari_active_contour_correction(img_path,
                                     gaussian_sigma=0.3,
                                     clahe_clip_limit=0.03,
                                     tophat_disk_size=15,
                                     snake_sigma=3,
                                     alpha=0.015,
                                     beta=10,
                                     w_line=0,
                                     w_edge=1,
                                     gamma=0.001,
                                     max_num_iter=2500,
                                     min_object_size=600):
    """
    Run active contour segmentation then open napari to correct the result.

    Usage
    -----
    viewer, labels_layer = napari_active_contour_correction(img_path)
    # In napari: use the paintbrush on the 'segmentation' layer to fix errors
    # When done:
    mask = np.array(labels_layer.data) > 0

    Returns
    -------
    viewer       : napari.Viewer
    labels_layer : napari Labels layer — edit with paintbrush, then access .data
    """
    import napari

    mask = chromosome_mask_with_active_contour(
        img_path,
        gaussian_sigma=gaussian_sigma,
        clahe_clip_limit=clahe_clip_limit,
        tophat_disk_size=tophat_disk_size,
        snake_sigma=snake_sigma,
        alpha=alpha, beta=beta,
        w_line=w_line, w_edge=w_edge,
        gamma=gamma, max_num_iter=max_num_iter,
        min_object_size=min_object_size,
        plot=False
    )

    if isinstance(img_path, (str, Path)):
        img_raw = skimage.io.imread(img_path, as_gray=True)
    else:
        img_raw = np.array(img_path, dtype=float)
    img = np.clip(skimage.filters.gaussian(img_raw, gaussian_sigma, preserve_range=True), 0, 1)

    labels_init = label(mask)

    viewer       = napari.Viewer()
    viewer.add_image(img, name='image', colormap='gray')
    labels_layer = viewer.add_labels(labels_init, name='segmentation')

    return viewer, labels_layer

def napari_contrast_stretching_correction(img_path,
                                          gaussian_sigma=0.3,
                                          tophat_disk_size=15,
                                          block_size=101,
                                          opening_disk_size=5,
                                          min_object_size=600,
                                          with_otsu=False):
    """
    Run contrast-stretching segmentation then open napari to correct the result.

    Usage
    -----
    viewer, labels_layer = napari_contrast_stretching_correction(img_path)
    # In napari: use the paintbrush on the 'segmentation' layer to fix errors
    # When done:
    mask = np.array(labels_layer.data) > 0

    Returns
    -------
    viewer       : napari.Viewer
    labels_layer : napari Labels layer — edit with paintbrush, then access .data
    """
    import napari

    mask = chromosome_mask_with_contrast_stretching(
        img_path,
        gaussian_sigma=gaussian_sigma,
        tophat_disk_size=tophat_disk_size,
        block_size=block_size,
        opening_disk_size=opening_disk_size,
        min_object_size=min_object_size,
        with_otsu=with_otsu,
        plot=False
    )

    if isinstance(img_path, (str, Path)):
        img_raw = skimage.io.imread(img_path, as_gray=True)
    else:
        img_raw = np.array(img_path, dtype=float)
    img = np.clip(skimage.filters.gaussian(img_raw, gaussian_sigma, preserve_range=True), 0, 1)

    labels_init = label(mask)

    viewer       = napari.Viewer()
    viewer.add_image(img, name='image', colormap='gray')
    labels_layer = viewer.add_labels(labels_init, name='segmentation')

    return viewer, labels_layer

def napari_adaptive_histogram_correction(img_path,
                                         gaussian_sigma=0.3,
                                         clahe_clip_limit=0.03,
                                         tophat_disk_size=15,
                                         block_size=101,
                                         opening_disk_size=5,
                                         min_object_size=600,
                                         min_distance=10,
                                         with_otsu=True):
    """
    Run adaptive histogram (CLAHE + watershed) segmentation then open napari to correct the result.

    Usage
    -----
    viewer, labels_layer = napari_adaptive_histogram_correction(img_path)
    # In napari: use the paintbrush on the 'segmentation' layer to fix errors
    # When done:
    mask = np.array(labels_layer.data) > 0

    Returns
    -------
    viewer       : napari.Viewer
    labels_layer : napari Labels layer — edit with paintbrush, then access .data
    """
    import napari

    mask = chromosome_mask_with_adaptive_histogram(
        img_path,
        gaussian_sigma=gaussian_sigma,
        clahe_clip_limit=clahe_clip_limit,
        tophat_disk_size=tophat_disk_size,
        block_size=block_size,
        opening_disk_size=opening_disk_size,
        min_object_size=min_object_size,
        min_distance=min_distance,
        with_otsu=with_otsu,
        plot=False
    )

    if isinstance(img_path, (str, Path)):
        img_raw = skimage.io.imread(img_path, as_gray=True)
    else:
        img_raw = np.array(img_path, dtype=float)
    img = np.clip(skimage.filters.gaussian(img_raw, gaussian_sigma, preserve_range=True), 0, 1)

    labels_init = label(mask)

    viewer       = napari.Viewer()
    viewer.add_image(img, name='image', colormap='gray')
    labels_layer = viewer.add_labels(labels_init, name='segmentation')

    return viewer, labels_layer


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

    gradient  = skimage.filters.sobel(img_adapteq)
    distance  = distance_transform_edt(binary_opened)
    coords    = peak_local_max(distance, min_distance=min_distance, labels=binary_opened)
    peak_mask = np.zeros(distance.shape, dtype=bool)
    if coords.size:
        peak_mask[tuple(coords.T)] = True
    markers   = label(peak_mask)
    labels_ws = watershed(gradient, markers, mask=binary_opened)

    import napari
    viewer       = napari.Viewer()
    viewer.add_image(img, name='image', colormap='gray')
    labels_layer = viewer.add_labels(labels_ws, name='segmentation')

    return viewer, labels_layer

def napari_semi_auto_seed_watershed(img_path,
                                    gaussian_sigma=0.3,
                                    clahe_clip_limit=0.03,
                                    tophat_disk_size=15,
                                    opening_disk_size=5,
                                    min_object_size=600,
                                    min_distance=10,
                                    with_otsu=True):
    """
    Semi-automated watershed with editable seeds in napari.

    Runs the full auto pipeline (CLAHE + top-hat + peak detection), displays
    the auto-detected seeds as a Points layer, then lets you correct them
    before running watershed.

    Usage
    -----
    viewer, points_layer, binary_mask = napari_semi_auto_seed_watershed(img_path)
    # In napari:
    #   - Press P to add missing seeds (one click per missed chromosome)
    #   - Select a point and press Delete to remove a bad seed
    #   - Drag points to reposition them
    # When happy, run Step 2:
    labels_ws = run_manual_seed_watershed(points_layer, binary_mask)

    Returns
    -------
    viewer       : napari.Viewer
    points_layer : napari Points layer — edit seeds, then pass to run_manual_seed_watershed
    binary_mask  : np.ndarray (bool)
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
        thresh = skimage.filters.threshold_local(tophat, 101, method='gaussian')

    binary_local  = tophat > thresh
    binary_opened = skimage.morphology.opening(binary_local, disk(opening_disk_size))
    binary_opened = remove_small_objects(binary_opened, min_size=min_object_size)
    binary_opened = clear_border(binary_opened)

    distance = distance_transform_edt(binary_opened)
    coords   = peak_local_max(distance, min_distance=min_distance, labels=binary_opened)

    import napari
    viewer = napari.Viewer()
    viewer.add_image(img, name='image', colormap='gray')
    viewer.add_image(binary_opened.astype(np.uint8), name='binary_mask',
                     colormap='green', blending='additive', opacity=0.3)
    points_layer = viewer.add_points(coords, name='seeds', ndim=2,
                                     size=10, face_color='red')

    return viewer, points_layer, binary_opened

def run_manual_seed_watershed(points_layer, binary_mask):
    """
    Step 2 — Run watershed using the seeds placed in napari.

    Parameters
    ----------
    points_layer : napari Points layer returned by napari_manual_seed_watershed
    binary_mask  : np.ndarray (bool) returned by napari_manual_seed_watershed

    Returns
    -------
    labels_ws : np.ndarray (int) — watershed label image
    """
    coords = np.array(points_layer.data).astype(int)
    if len(coords) == 0:
        raise ValueError("No seeds found. Place at least one point in the 'seeds' layer.")

    # clip coordinates to image bounds
    coords[:, 0] = np.clip(coords[:, 0], 0, binary_mask.shape[0] - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, binary_mask.shape[1] - 1)

    markers = np.zeros(binary_mask.shape, dtype=int)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    distance = distance_transform_edt(binary_mask)
    labels_ws = watershed(-distance, markers, mask=binary_mask)

    print(f"Watershed complete: {len(coords)} seeds → {labels_ws.max()} labeled regions")
    return labels_ws


# ---------------------------------------------------------------------------
# Denver group classifier
# ---------------------------------------------------------------------------

# Denver system (1960) groups human chromosomes by size and centromere position:
#
#   Group A  —  chr 1, 2, 3      : largest;       metacentric (1,3) / submetacentric (2)
#   Group B  —  chr 4, 5         : large;          submetacentric
#   Group C  —  chr 6–12, X      : medium;         submetacentric
#   Group D  —  chr 13, 14, 15   : medium;         acrocentric (with satellites)
#   Group E  —  chr 16, 17, 18   : smaller;        metacentric (16) / submetacentric (17,18)
#   Group F  —  chr 19, 20       : small;          metacentric
#   Group G  —  chr 21, 22, Y    : very small;     acrocentric (with satellites)
#
# Classification uses two morphometric features derived from the segmented masks:
#   1. relative_length  : chromosome length as % of the total detected length in the image
#                         (serves as a proxy for absolute size rank)
#   2. centromeric_index: short_arm / total_length  (0 = acrocentric, 0.5 = metacentric)
#      — requires a detected centromere_position (row of minimum width in aligned crop)

# Centromere type thresholds (standard Denver/Paris convention):
#   metacentric    : CI >= 0.38
#   submetacentric : 0.25 <= CI < 0.38
#   acrocentric    : CI < 0.25

# Relative-length thresholds are calibrated for a diploid spread (~46 chromosomes).
# When fewer chromosomes are detected the relative lengths inflate proportionally,
# so an `n_detected` correction factor is applied.

_DENVER_META   = 0.38   # CI >= this → metacentric
_DENVER_SUBMETA = 0.25  # CI >= this → submetacentric  (else acrocentric)

# Expected number of chromosomes in a diploid human cell
_DIPLOID_N = 46


def _centromere_type(ci):
    """Return centromere type string from centromeric index."""
    if pd.isna(ci):
        return 'unknown'
    if ci >= _DENVER_META:
        return 'metacentric'
    if ci >= _DENVER_SUBMETA:
        return 'submetacentric'
    return 'acrocentric'


def _denver_rule(rel_len_corrected, ci):
    """
    Map (corrected relative length, centromeric index) → Denver group A–G.

    rel_len_corrected : relative_length scaled as if all 46 chromosomes were detected
    ci                : centromeric_index (NaN allowed — falls back to size only)
    """
    is_meta   = (not pd.isna(ci)) and ci >= _DENVER_META
    is_submeta = (not pd.isna(ci)) and (_DENVER_SUBMETA <= ci < _DENVER_META)
    is_acro   = (not pd.isna(ci)) and ci < _DENVER_SUBMETA
    ci_known  = not pd.isna(ci)

    # Group A: chromosomes 1, 2, 3 — largest, meta/submeta
    if rel_len_corrected > 3.5:
        return 'A'

    # Group B: chromosomes 4, 5 — large, submetacentric
    if rel_len_corrected > 2.8:
        if ci_known:
            return 'B' if (is_submeta or is_acro) else 'A'
        return 'B'

    # Group C: chromosomes 6–12, X — medium, submetacentric
    if rel_len_corrected > 1.8:
        if ci_known:
            if is_acro:
                return 'D'    # acrocentric at this size → Group D
            return 'C'
        return 'C'

    # Groups D and E overlap in size; centromere type is the key discriminator
    if rel_len_corrected > 1.3:
        if ci_known:
            return 'D' if is_acro else 'E'
        return 'D/E'   # ambiguous without centromere info

    # Group F: chromosomes 19, 20 — small, metacentric
    if rel_len_corrected > 1.0:
        if ci_known:
            return 'F' if (is_meta or is_submeta) else 'G'
        return 'F'

    # Group G: chromosomes 21, 22, Y — very small, acrocentric
    return 'G'


def classify_denver_groups(df,
                           length_col='length',
                           centromere_col='centromere_position',
                           image_col='image_name'):
    """
    Classify detected chromosomes into Denver groups A–G.

    Parameters
    ----------
    df              : pd.DataFrame — one row per detected chromosome, as produced
                      by the notebook's per-chromosome feature extraction loop.
    length_col      : str — column containing chromosome length in pixels
                      (major_axis_length from regionprops).
    centromere_col  : str — column containing the centromere row position within
                      the aligned chromosome crop (from find_centromere).
                      May contain NaN when centromere detection fails.
    image_col       : str — column identifying the source image; used to compute
                      relative lengths within each karyotype spread.

    Returns
    -------
    pd.DataFrame — copy of df with four additional columns:

        relative_length   : chromosome length as % of total detected length in
                            the same image (higher = larger chromosome)
        centromeric_index      : short_arm / total_length, in [0, 0.5]
                                 NaN when centromere_position is NaN
        centromeric_index_area : min(p_arm_pixels, q_arm_pixels) / total_pixels
                                 area-based CI from splitting chrom_img at the
                                 centromere row; captures arm width variation.
                                 NaN when centromere_position or chrom_img is NaN.
        centromere_type   : 'metacentric' | 'submetacentric' | 'acrocentric' | 'unknown'
        arm_ratio         : longer_arm / shorter_arm, in [1, ∞)
                            NaN when centromere_position is NaN
        denver_group      : 'A'–'G', 'D/E' (ambiguous without centromere), or 'unknown'

    Notes
    -----
    - Classification is based on morphometrics alone (size + centromere position).
      Without G-banding, chromosomes within the same Denver group cannot be
      individually identified.
    - The relative-length thresholds assume a diploid spread.  When significantly
      fewer than 46 chromosomes are detected (e.g. due to occlusion), a correction
      factor rescales the relative lengths before applying the rules.
    - Chromosomes that are very small and lack centromere information are assigned
      Group G by default (smallest size bin).

    Examples
    --------
    >>> df_classified = classify_denver_groups(df)
    >>> df_classified[['image_name', 'length', 'relative_length',
    ...                 'centromeric_index', 'centromere_type', 'denver_group']].head()
    """
    result = df.copy()

    # ------------------------------------------------------------------ #
    # 1. Centromeric index  (short_arm / total_length)
    # ------------------------------------------------------------------ #
    def _ci(row):
        cp = row[centromere_col]
        ln = row['aligned_length'] if 'aligned_length' in row.index else row[length_col]
        if pd.isna(cp) or pd.isna(ln) or ln == 0:
            return np.nan
        p = float(cp)
        q = float(ln) - p
        if p <= 0 or q <= 0:
            return np.nan
        short = min(p, q)
        return short / float(ln)

    result['centromeric_index'] = result.apply(_ci, axis=1)

    # ------------------------------------------------------------------ #
    # 1b. Area-based centromeric index  CI(A) = min(Ap, Aq) / (Ap + Aq)
    #     Splits chrom_img at the centromere row and counts nonzero pixels
    #     in each arm, capturing width variation along the chromosome.
    # ------------------------------------------------------------------ #
    def _ci_area(row):
        img = row['chrom_img']
        cp  = row[centromere_col] #centromere position (row index within chrom_img)
        if pd.isna(cp) or img is None:
            return np.nan
        cp = int(round(float(cp)))
        p_area = int(np.count_nonzero(img[:cp]))
        q_area = int(np.count_nonzero(img[cp:]))
        total  = p_area + q_area
        if total == 0:
            return np.nan
        return min(p_area, q_area) / total

    result['centromeric_index_area'] = result.apply(_ci_area, axis=1)
    result['centromere_type']        = result['centromeric_index'].map(_centromere_type)

    # ------------------------------------------------------------------ #
    # 1c. Arm ratio  (longer_arm / shorter_arm)
    # ------------------------------------------------------------------ #
    def _arm_ratio(row):
        cp = row[centromere_col]
        # use aligned_length if available — same coordinate system as centromere_position
        ln = row['aligned_length'] if 'aligned_length' in row.index else row[length_col]
        if pd.isna(cp) or pd.isna(ln) or ln == 0:
            return np.nan
        p = float(cp)
        q = float(ln) - p
        if p <= 0 or q <= 0:
            return np.nan
        return max(p, q) / min(p, q)

    result['arm_ratio'] = result.apply(_arm_ratio, axis=1)

    # ------------------------------------------------------------------ #
    # 2. Relative length within each image (% of total detected length)
    # ------------------------------------------------------------------ #
    result['relative_length'] = result.groupby(image_col)[length_col].transform(
        lambda x: x / x.sum() * 100
    )

    # ------------------------------------------------------------------ #
    # 3. Correction factor for partial detection
    #    If only k < 46 chromosomes detected, relative lengths are inflated
    #    by 46/k.  Rescale so thresholds remain valid.
    # ------------------------------------------------------------------ #
    n_per_image = result.groupby(image_col)[length_col].transform('count')
    correction  = n_per_image / _DIPLOID_N          # < 1 when fewer detected
    result['_rl_corrected'] = result['relative_length'] * correction


    result['denver_group'] = result.apply(
        lambda r: _denver_rule(r['_rl_corrected'], r['centromeric_index']),
        axis=1
    )
    result.drop(columns=['_rl_corrected'], inplace=True)

    return result


def extract_chromosome_features(img_paths, method='adaptive_histogram',
                                 pad=5, plot=False):
    """
    Segment chromosomes from a list of karyotype images and extract
    per-chromosome morphometric features into a DataFrame.

    Parameters
    ----------
    img_paths : list of Path — karyotype image file paths to process
    method    : str — segmentation method used; stored in the 'method' column.
                      'adaptive_histogram' | 'contrast_stretching'
    pad       : int — pixel padding added around each chromosome bounding box
    plot      : bool — if True, passes plot=True to the segmentation function

    Returns
    -------
    pd.DataFrame — one row per detected chromosome with columns:
        image_name, chromosome_label, chrom_img, method,
        area, length, width, eccentricity,
        centroid_row, centroid_col, orientation,
        bbox_minr, bbox_minc, bbox_maxr, bbox_maxc,
        number_of_chromosomes, solidity
    """
    records = []

    for img_path in img_paths:
        img_gray = skimage.io.imread(img_path, as_gray=True)
        img_gray = np.clip(img_gray, 0, 1)

        mask    = chromosome_mask_with_adaptive_histogram(img_path, plot=plot)
        # mask = chromosome_mask_thresholding(img_path, plot=plot)
        labeled = label(mask)
        regions = regionprops(labeled)

        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            minr = max(0, minr - pad)
            minc = max(0, minc - pad)
            maxr = min(mask.shape[0], maxr + pad)
            maxc = min(mask.shape[1], maxc + pad)

            chrom_mask = (labeled[minr:maxr, minc:maxc] == region.label)
            img_crop   = img_gray[minr:maxr, minc:maxc]
            chrom_img  = img_crop * chrom_mask

            records.append({
                'image_name':             img_path.name,
                'chromosome_label':       region.label,
                'chrom_img':              chrom_img,
                'method':                 method,
                'area':                   region.area,
                'length':                 region.major_axis_length,
                'width':                  region.minor_axis_length,
                'eccentricity':           region.eccentricity,
                'centroid_row':           region.centroid[0],
                'centroid_col':           region.centroid[1],
                'orientation':            region.orientation,
                'bbox_minr':              region.bbox[0],
                'bbox_minc':              region.bbox[1],
                'bbox_maxr':              region.bbox[2],
                'bbox_maxc':              region.bbox[3],
                'number_of_chromosomes':  labeled.max(),
                'solidity':               region.solidity,
            })

    return pd.DataFrame(records)


def extract_centromere_and_bands(df, plot=False):
    """
    Align each chromosome, detect its centromere position and band count,
    and store the results back into the DataFrame.

    Parameters
    ----------
    df   : pd.DataFrame — one row per chromosome with columns:
               chrom_img, orientation, method, image_name, chromosome_label
    plot : bool — if True, display alignment + intensity profile + arm ratio
                  for each chromosome (default False)

    Returns
    -------
    pd.DataFrame — copy of df with two new columns:
        centromere_position : int row index of centromere in aligned image,
                              or NaN if detection failed
        n_bands             : number of dark bands detected, or NaN on failure
    """
    from helper_functions import find_centromere, extract_intensity_profile, detect_bands

    result               = df.copy()
    centromere_positions = []
    n_bands_list         = []
    aligned_lengths      = []

    for _, row in result.iterrows():
        chrom = row['chrom_img']

        # align — long axis vertical
        angle   = -row['orientation'] * 180 / np.pi
        aligned = skimage.transform.rotate(chrom, angle, resize=True, cval=0,
                                           preserve_range=True)

        # fix background: black (0) → white (1)
        aligned_white = np.where(aligned == 0, 1.0, aligned)

        # apply same contrast enhancement used during segmentation
        if row['method'] == 'adaptive_histogram':
            display_img = exposure.equalize_adapthist(aligned_white, clip_limit=0.03)
        else:
            p2, p98     = np.percentile(aligned_white[aligned_white < 1.0], (2, 98))
            display_img = exposure.rescale_intensity(aligned_white, in_range=(p2, p98))

        # store aligned image height — used as the true chromosome length
        # for arm ratio / centromeric index (avoids mismatch with regionprops length)
        aligned_lengths.append(aligned_white.shape[0])

        # centromere
        try:
            c_pos = find_centromere(aligned_white)
            centromere_positions.append(c_pos)
        except Exception:
            c_pos = np.nan
            centromere_positions.append(np.nan)

        # bands
        try:
            profile            = extract_intensity_profile(aligned_white)
            smoothed, peaks, _ = detect_bands(profile)
            n_bands_list.append(len(peaks))
        except Exception:
            profile, smoothed, peaks = None, None, []
            n_bands_list.append(np.nan)

        if plot:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(display_img, cmap='gray', vmin=0, vmax=1)
            if not np.isnan(c_pos):
                axes[0].axhline(c_pos, color='red', lw=1.5,
                                label=f'centromere row {int(c_pos)}')
                axes[0].legend(fontsize=7)
            axes[0].set_title(
                f"{row['image_name']} | chrom {row['chromosome_label']} | {row['method']}")
            axes[0].axis('off')

            if profile is not None:
                axes[1].plot(profile,  color='steelblue', lw=1,   label='raw')
                axes[1].plot(smoothed, color='orange',    lw=1.5, label='smoothed')
                # if len(peaks):
                #     axes[1].plot(peaks, smoothed[peaks], 'rv', markersize=6,
                #                  label=f'{len(peaks)} bands')
                if not np.isnan(c_pos):
                    axes[1].axvline(c_pos, color='red', lw=1.2, linestyle='--',
                                    label='centromere')
                axes[1].set_title('Intensity profile')
                axes[1].set_xlabel('Row')
                axes[1].legend(fontsize=7)

            if not np.isnan(c_pos):
                total = aligned_white.shape[0]
                p_arm = c_pos
                q_arm = total - c_pos
                ar    = p_arm / q_arm if q_arm > 0 else np.nan
                axes[2].bar(['p arm', 'q arm'], [p_arm, q_arm],
                            color=['steelblue', 'coral'])
                axes[2].set_title(f'Arm ratio p/q = {ar:.2f}')
            else:
                axes[2].set_title('Arm ratio: centromere not detected')

            plt.tight_layout()
            plt.show()

    result['centromere_position'] = centromere_positions
    result['n_bands']             = n_bands_list
    result['aligned_length']      = aligned_lengths
    return result


def plot_segmentation_comparison(img_paths, overlay_color=(0.2, 0.8, 0.2), alpha=0.4,
                                  figsize=(18, 4)):
    """
    For each image, run all four segmentation algorithms, overlay the resulting
    mask in colour on the original image, and display them side by side.

    Parameters
    ----------
    img_paths      : list of Path — images to process
    overlay_color  : RGB tuple (0–1) — colour used for the mask overlay
    alpha          : float — transparency of the overlay (0 = invisible, 1 = opaque)
    figsize        : tuple — figure size per image row

    Returns
    -------
    None — displays matplotlib figures
    """
    algorithms = {
        'Original':             None,
        'Adaptive Histogram':   lambda p: chromosome_mask_with_adaptive_histogram(p, plot=False),
        'Contrast Stretching':  lambda p: chromosome_mask_with_contrast_stretching(p, plot=False),
        'Thresholding':         lambda p: chromosome_mask_thresholding(p, plot=False),
        'Active Contour':       lambda p: chromosome_mask_with_active_contour(p, plot=False),
    }

    for img_path in img_paths:
        img_gray = skimage.io.imread(img_path, as_gray=True)
        img_gray = np.clip(img_gray, 0, 1)

        # convert grayscale to RGB for overlay
        img_rgb  = np.stack([img_gray] * 3, axis=-1)

        fig, axes = plt.subplots(1, len(algorithms), figsize=figsize)
        fig.suptitle(Path(img_path).name, fontsize=11)

        for ax, (name, fn) in zip(axes, algorithms.items()):
            if fn is None:
                ax.imshow(img_rgb)
                ax.set_title('Original', fontsize=9)
            else:
                try:
                    mask = fn(img_path)
                    overlay = img_rgb.copy()
                    for c, val in enumerate(overlay_color):
                        overlay[:, :, c] = np.where(mask,
                                                     alpha * val + (1 - alpha) * img_rgb[:, :, c],
                                                     img_rgb[:, :, c])
                    ax.imshow(overlay)
                    n = label(mask).max()
                    ax.set_title(f'{name}\n({n} detected)', fontsize=9)
                except Exception as e:
                    ax.imshow(img_rgb)
                    ax.set_title(f'{name}\n(failed)', fontsize=9, color='red')

            ax.axis('off')

        plt.tight_layout()
        plt.show()


def plot_denver_groups(df_classified, image_col='image_name', figsize=(10, 5)):
    """
    Bar chart showing the count of chromosomes per Denver group.

    Parameters
    ----------
    df_classified : pd.DataFrame — output of classify_denver_groups()
    image_col     : str — if provided, one subplot per unique image; if None, aggregate
    figsize       : tuple

    Returns
    -------
    fig, axes
    """
    groups      = ['A', 'B', 'C', 'D', 'D/E', 'E', 'F', 'G', 'unknown']
    group_color = {'A': '#e41a1c', 'B': '#ff7f00', 'C': '#4daf4a',
                   'D': '#377eb8', 'D/E': '#984ea3', 'E': '#a65628',
                   'F': '#f781bf', 'G': '#999999', 'unknown': '#dddddd'}

    images = df_classified[image_col].unique() if image_col else [None]
    ncols  = min(len(images), 3)
    nrows  = int(np.ceil(len(images) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize[0] * ncols, figsize[1] * nrows),
                             squeeze=False)

    for idx, img in enumerate(images):
        ax  = axes[idx // ncols][idx % ncols]
        sub = df_classified[df_classified[image_col] == img] if img else df_classified
        counts = sub['denver_group'].value_counts().reindex(groups, fill_value=0)
        colors = [group_color.get(g, '#dddddd') for g in counts.index]
        ax.bar(counts.index, counts.values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(str(img) if img else 'All images', fontsize=8)
        ax.set_xlabel('Denver group')
        ax.set_ylabel('Count')

    # Hide unused subplots
    for idx in range(len(images), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    return fig, axes