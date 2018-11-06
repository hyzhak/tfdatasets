import logging
import os
import tarfile
from tenacity import after_log, retry
import urllib

logger = logging.getLogger(__name__)


@retry(after=after_log(logger, logging.DEBUG))
def urlretrieve_with_retry(url, filename=None):
    return urllib.request.urlretrieve(url, filename)


def maybe_download(filename, work_directory, source_url):
    """Download the data from source url, unless it's already here.

    Copied and adapted from tensorflow/contrib/learn/python/learn/datasets/base.py
    because original method was depricated

    Args:
        filename: string, name of the file in the directory.
        work_directory: string, path to working directory.
        source_url: url to download from if file doesn't exist.

    Returns:
        Path to resulting file.
    """
    if not os.path.exists(work_directory):
        os.makedirs(work_directory)

    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        temp_file_name, _ = urlretrieve_with_retry(source_url, filepath)
        # shutil.copyfile(temp_file_name, filepath)
        size = os.path.getsize(filepath)
        logger.info('Successfully downloaded', filename, str(size), 'bytes.')
    return filepath


def download_and_extract(data_dir, download_url, filename):
    """
    download and extract (if needed)
    :param data_dir:
    :param download_url:
    :param filename:
    :return:
    """
    # download dataset if not already downloaded
    maybe_download(filename, data_dir, download_url)
    _, ext = os.path.splitext(filename)
    if ext == '.gz':
        # for reading with gzip compression
        tarfile.open(os.path.join(data_dir, filename),
                     'r:gz').extractall(data_dir)
    elif ext == '.bz2':
        # for reading with bzip2 compression
        tarfile.open(os.path.join(data_dir, filename),
                     'r:bz2').extractall(data_dir)
    elif ext == '.xz':
        # for reading with lzma compression
        tarfile.open(os.path.join(data_dir, filename),
                     'r:xz').extractall(data_dir)


__all__ = [
    download_and_extract
]
