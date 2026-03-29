from pathlib import Path
from skimage import exposure, io
from skimage import data, img_as_float
import matplotlib
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skimage 
from skimage.filters import threshold_otsu
from scipy.signal import savgol_filter, find_peaks

matplotlib.rcParams['font.size'] = 8
from algorithms import *

def display_image_histogram(img_input, bins=256, title=None,
                            mask_background=True, log_scale=False):
    """
    Display an image alongside its intensity histogram and CDF,
    with key statistics annotated.

    Parameters
    ----------
    img_input        : str, Path, or 2D array
    bins             : int   — histogram bins (default 256)
    title            : str   — custom title (default: filename stem or "Image")
    mask_background  : bool  — exclude background pixels via Otsu before
                               computing histogram/stats (default True).
                               Useful for chromosome crops where the white
                               background dominates the histogram.
    log_scale        : bool  — log y-axis on histogram (default False)

    Returns
    -------
    stats : dict — mean, std, median, p2, p98, dynamic_range
    """
    if isinstance(img_input, (str, Path)):
        img = skimage.io.imread(img_input, as_gray=True)
        label = Path(img_input).stem if title is None else title
    else:
        img = np.asarray(img_input, dtype=float)
        label = title or "Image"

    img = np.clip(img, 0, 1)


    if mask_background:
        thresh  = threshold_otsu(img)
        fg_mask = img < thresh         
        pixels  = img[fg_mask]
        mask_note = f"foreground only (Otsu < {thresh:.2f})"
    else:
        pixels    = img.ravel()
        mask_note = "all pixels"

    if pixels.size == 0:           
        pixels    = img.ravel()
        mask_note = "all pixels (fallback)"

    mean   = pixels.mean()
    std    = pixels.std()
    median = np.median(pixels)
    p2, p98 = np.percentile(pixels, (2, 98))

    stats = dict(mean=mean, std=std, median=median,
                 p2=p2, p98=p98, dynamic_range=p98 - p2)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4),
                             gridspec_kw={"width_ratios": [1.2, 1.5, 1]})


    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(label, fontsize=9)
    axes[0].axis("off")

   
    axes[1].hist(pixels, bins=bins, color="steelblue",
                 density=True, linewidth=0)
    axes[1].axvline(mean,   color="red",    lw=1.2, linestyle="-",
                    label=f"mean   {mean:.3f}")
    axes[1].axvline(median, color="orange", lw=1.2, linestyle="--",
                    label=f"median {median:.3f}")
    axes[1].axvline(p2,     color="gray",   lw=1.0, linestyle=":",
                    label=f"p2     {p2:.3f}")
    axes[1].axvline(p98,    color="gray",   lw=1.0, linestyle=":",
                    label=f"p98    {p98:.3f}")
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Pixel intensity", fontsize=9)
    axes[1].set_ylabel("Density",         fontsize=9)
    axes[1].set_title(f"Histogram ({mask_note})", fontsize=8)
    axes[1].legend(fontsize=7, loc="upper left")
    if log_scale:
        axes[1].set_yscale("log")


    sorted_px  = np.sort(pixels)
    cdf_vals   = np.arange(1, len(sorted_px) + 1) / len(sorted_px)
    axes[2].plot(sorted_px, cdf_vals, color="steelblue", lw=1.5)
    axes[2].axhline(0.02, color="gray", lw=0.8, linestyle=":")
    axes[2].axhline(0.98, color="gray", lw=0.8, linestyle=":")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Intensity", fontsize=9)
    axes[2].set_ylabel("CDF",       fontsize=9)
    axes[2].set_title("CDF",        fontsize=9)

    fig.suptitle(
        f"std={std:.3f}   dynamic range (p2–p98)={p98 - p2:.3f}",
        fontsize=9, y=1.01
    )
    plt.tight_layout()
    plt.show()

    return stats

def plot_img_and_hist(image, bins=256):
    """Plot an image along with its histogram and cumulative histogram."""
    if isinstance(image, (str, Path)):
        image = io.imread(image)
    image = img_as_float(image)
    fig, (ax_img, ax_hist) = plt.subplots(1, 2)
    ax_cdf = ax_hist.twinx()

    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return fig, ax_img, ax_hist, ax_cdf

def compute_image_quality_metrics(image_path):

    img = skimage.io.imread(image_path)

    if img.ndim == 3:
        gray = skimage.color.rgb2gray(img)
    else:
        gray = img / 255.0

    metrics = {}

    metrics["contrast"] = np.std(gray)

    laplacian = skimage.filters.laplace(gray)
    metrics["sharpness"] = laplacian.var()

    mean_signal = np.mean(gray)
    noise = np.std(gray)

    metrics["SNR"] = mean_signal / noise if noise != 0 else 0

    block_size = (32,32)

    h, w = gray.shape
    h = h - (h % 32)
    w = w - (w % 32)

    cropped = gray[:h, :w]

    blocks = skimage.util.view_as_blocks(cropped, block_size)

    block_means = blocks.mean(axis=(2,3))
    metrics["background_uniformity"] = np.std(block_means)

    hist, _ = np.histogram(gray, bins=256, range=(0,1))
    metrics["histogram_spread"] = np.std(hist)

    windows = skimage.util.view_as_windows(gray, (15,15))
    local_std_map = np.std(windows, axis=(2,3))

    metrics["local_std_variation"] = np.std(local_std_map)

    return metrics

def plot_segmentation_grid(img_paths, segment_fn, n_cols=6, figsize=(18, 10), **kwargs):
    """
    Plot a 3-row diagnostic grid for a list of images using a segmentation function.

    Rows: Preprocessed | Binary mask | Overlay

    Parameters
    ----------
    img_paths    : list — list of file paths or arrays
    segment_fn   : callable — your segment_chromosomes or segment_chromosomes_v2
    n_cols       : int — how many images to show (default 6)
    figsize      : tuple — figure size
    **kwargs     : extra args passed to segment_fn (e.g. gaussian_sigma=0.5)
    """

    row_labels = ["Preprocessed", "Binary mask", "Overlay"]
    img_paths  = img_paths[:n_cols]

    fig, axes = plt.subplots(3, len(img_paths), figsize=figsize)

    for col, img_path in enumerate(img_paths):

        if isinstance(img_path, (str, Path)):
            img_raw = skimage.io.imread(img_path, as_gray=True)
        else:
            img_raw = img_path.copy()

        img = np.clip(skimage.filters.gaussian(img_raw, kwargs.get('gaussian_sigma', 0.3),
                                               preserve_range=True), 0, 1)

        mask = segment_fn(img_path, plot=False, **kwargs)

        overlay = np.stack([img, img, img], axis=-1)
        overlay[mask] = np.clip(overlay[mask] + [0.4, -0.1, -0.1], 0, 1)

        for row, (im, cmap) in enumerate([
            (img,     "gray"),
            (mask,    "gray"),
            (overlay, None),
        ]):
            ax = axes[row, col]
            ax.imshow(im, cmap=cmap, vmin=0, vmax=1)
            ax.axis("off")

            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=9)
            if row == 0:
                title = Path(img_path).stem if isinstance(img_path, (str, Path)) else f"img_{col}"
                ax.set_title(title, fontsize=8)

    plt.suptitle("Chromosome segmentation pipeline", fontsize=13)
    plt.tight_layout()
    plt.show()

def extract_intensity_profile(chrom_img):
    """
    Compute a 1D mean intensity profile along the long axis of an aligned chromosome.
    The image is inverted internally so dark bands appear as peaks.

    Parameters
    ----------
    chrom_img : 2D array — grayscale crop, long axis along rows (already rotated)

    Returns
    -------
    profile : 1D array, length == chrom_img.shape[0]
    """
    p2, p98 = np.percentile(chrom_img, (2, 98))
    chrom_img = exposure.rescale_intensity(chrom_img, in_range=(p2, p98))
    # Mask out the white background so only chromosome pixels contribute
    thresh     = threshold_otsu(chrom_img)
    chrom_mask = chrom_img < thresh          # chromosome pixels are darker

    inv     = 1.0 - chrom_img               # invert: dark bands → high values
    profile = np.zeros(chrom_img.shape[0])

    for r in range(chrom_img.shape[0]):
        cols = chrom_mask[r]
        if cols.sum() > 0:
            profile[r] = inv[r, cols].mean()

    return profile

def detect_bands(profile, window=11, poly=3, prominence=0.02, min_width=2):
    """
    Smooth a 1D intensity profile and detect dark band peaks.

    Parameters
    ----------
    profile     : 1D array from extract_intensity_profile
    window      : int  — Savitzky-Golay window length (must be odd, default 11)
    poly        : int  — Savitzky-Golay polynomial order (default 3)
    prominence  : float — minimum peak prominence to count as a band (default 0.02)
    min_width   : int  — minimum peak width in pixels (default 2)

    Returns
    -------
    smoothed : 1D array — smoothed profile
    peaks    : int array — row indices of detected band centres
    props    : dict — find_peaks properties (widths, prominences, etc.)
    """
    smoothed        = savgol_filter(profile, window_length=window, polyorder=poly)
    peaks, props    = find_peaks(smoothed, prominence=prominence, width=min_width)
    return smoothed, peaks, props

def find_centromere(chrom_img):
    """
    Estimate the centromere position as the row of minimum chromosome width,
    searching only in the middle 50% of the chromosome to avoid the tips.

    Parameters
    ----------
    chrom_img : 2D array — grayscale crop (long axis along rows)

    Returns
    -------
    centromere_row : int
    """
    thresh       = threshold_otsu(chrom_img)
    chrom_mask   = chrom_img < thresh
    widths       = chrom_mask.sum(axis=1).astype(float)

    # Smooth the width profile to suppress noise
    widths_smooth = savgol_filter(widths, window_length=11, polyorder=3)

    # Restrict search to middle 50%
    lo           = len(widths) // 4
    hi           = 3 * len(widths) // 4
    centromere   = int(np.argmin(widths_smooth[lo:hi]) + lo)

    return centromere

def auto_segment_chromosomes(img_path,
                             gaussian_sigma=0.3,
                             clahe_clip_limit=0.03,
                             tophat_disk_size=15,
                             block_size=121,
                             closing_disk_size=3,
                             opening_disk_size=8,
                             min_object_size=600,
                             max_object_size=5000,
                             max_eccentricity=0.97,
                             min_distance=12,
                             spatial_blocks=9,
                             contrast_threshold=0.1,
                             uniformity_threshold=0.5,
                             plot=True):
    
    if isinstance(img_path, (str, Path)):
        img_raw = skimage.io.imread(img_path, as_gray=True)
    else:
        img_raw = np.array(img_path, dtype=float)

    img = skimage.filters.gaussian(img_raw, gaussian_sigma, preserve_range=True)
    img = np.clip(img, 0, 1)

    p1, p99        = np.percentile(img, [1, 99])
    global_contrast = p99 - p1

    h, w  = img.shape
    bh, bw = h // spatial_blocks, w // spatial_blocks
    block_stds = []
    for i in range(spatial_blocks):
        for j in range(spatial_blocks):
            block = img[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            block_stds.append(block.std())
    block_stds      = np.array(block_stds)
    spatial_cv      = block_stds.std() / (block_stds.mean() + 1e-8)

    poor_contrast    = global_contrast < contrast_threshold
    spatially_uneven = spatial_cv     > uniformity_threshold

    if poor_contrast and spatially_uneven:
        method = 'adaptive_histogram'
    else:
        method = 'contrast_stretching'

    print(f"global contrast: {global_contrast:.3f} | spatial CV: {spatial_cv:.3f} | → {method}")

    if method == 'adaptive_histogram':
        mask = chromosome_mask_with_adaptive_histogram(
            img_path,
            gaussian_sigma=gaussian_sigma,
            clahe_clip_limit=clahe_clip_limit,
            tophat_disk_size=tophat_disk_size,
            block_size=block_size,
            closing_disk_size=closing_disk_size,
            opening_disk_size=opening_disk_size,
            min_object_size=min_object_size,
            max_object_size=max_object_size,
            max_eccentricity=max_eccentricity,
            min_distance=min_distance,
            plot=plot
        )
    else:
        mask = chromosome_mask_with_contrast_stretching(
            img_path,
            gaussian_sigma=gaussian_sigma,
            tophat_disk_size=tophat_disk_size,
            block_size=block_size,
            closing_disk_size=closing_disk_size,
            opening_disk_size=opening_disk_size,
            min_object_size=min_object_size,
            max_object_size=max_object_size,
            max_eccentricity=max_eccentricity,
            plot=plot
        )

    return mask, method