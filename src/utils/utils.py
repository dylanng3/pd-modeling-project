import os
import random
import time
import numpy as np
import warnings
from tqdm import tqdm
from contextlib import contextmanager

# Suppress unnecessary warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Set environment variable to ensure UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Set seed for reproducibility
def set_seed(seed=42):
    """Set seed cho tất cả các thư viện ngẫu nhiên"""
    random.seed(seed)
    np.random.seed(seed)

# Basic utility functions
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print(f"{title} - done in {time.time() - t0:.0f}s")

def create_progress_bar(iterable, desc, total=None):
    """Tạo progress bar với tqdm"""
    return tqdm(iterable, desc=desc, total=total, ncols=100, 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')