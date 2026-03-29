from pathlib import Path
from skimage import exposure
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import skimage
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
                            tophat_disk_size=3,
                            block_size=101,
                            closing_disk_size=8,
                            opening_disk_size=5,
                            min_object_size=600,
                            max_object_size=30000,
                            max_eccentricity=0.97,
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
                        closing_disk_size=8,
                        opening_disk_size=8,
                        min_object_size=600,
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


    img_adapteq = exposure.equalize_adapthist(img, clip_limit=clahe_clip_limit)


    inverted = 1.0 - img_adapteq

    
    tophat = white_tophat(inverted, disk(tophat_disk_size))

    if with_otsu:

        local_thresholds =  threshold_otsu(tophat)

    else:

        local_thresholds = skimage.filters.threshold_local(tophat, block_size, method='gaussian')


    binary_local = tophat > local_thresholds

    # Remove large blobs (e.g. nuclei) before closing so they don't get fragmented
    labeled_pre = label(binary_local)
    for region in regionprops(labeled_pre):
        if region.area > max_object_size:
            binary_local[labeled_pre == region.label] = False

    # Close gaps between chromosome bands before opening
    # binary_closed = skimage.morphology.closing(binary_local, disk(closing_disk_size))
    binary_opened = skimage.morphology.opening(binary_local, disk(opening_disk_size))
    binary_opened = remove_small_objects(binary_opened, min_size=min_object_size)
    binary_opened = clear_border(binary_opened)

    # Watershed to split clustered chromosomes
    # Use gradient magnitude as the flooding surface so splits happen at true
    # intensity valleys (chromosome boundaries) rather than geometric midpoints.
    gradient  = skimage.filters.sobel(img_adapteq)
    distance  = distance_transform_edt(binary_opened)
    coords    = peak_local_max(distance, min_distance=min_distance, labels=binary_opened)
    peak_mask = np.zeros(distance.shape, dtype=bool)
    peak_mask[tuple(coords.T)] = True
    markers   = label(peak_mask)
    labels_ws = watershed(gradient, markers, mask=binary_opened)
    mask = labels_ws > 0

    # Remove remaining large objects and scale-bar-like artifacts (high eccentricity)
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
                                     max_num_iter=10,
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