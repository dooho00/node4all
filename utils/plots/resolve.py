from __future__ import annotations

import os
from typing import Optional

from .cfg import get_plot_dir


def resolve_plot_base(plot_root: Optional[str], *, plot_dir: Optional[str] = None) -> Optional[str]:
    if not plot_root:
        return None
    base_name = os.path.splitext(os.path.basename(str(plot_root)))[0]
    if not base_name:
        return None
    if plot_dir is None:
        plot_dir = get_plot_dir()
    if plot_dir:
        return os.path.join(str(plot_dir), base_name)
    return base_name
