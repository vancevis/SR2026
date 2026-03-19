from .diff_gaussian_rasterizer import DiffGaussianRenderer
from .gaussian_batch_renderer import GaussianBatchRenderer


class DiffGaussianBatchRenderer(DiffGaussianRenderer, GaussianBatchRenderer):
    """Combined renderer with single-view and batch rendering support."""
    pass


__all__ = ["DiffGaussianRenderer", "GaussianBatchRenderer", "DiffGaussianBatchRenderer"]
