import scipy.sparse as sp
import sklearn.utils
from sklearn.utils.validation import _num_samples


class Dataset(object):
    """Class for generating data mini-batches

    Parameters
    ----------
    *arrays
        Variable length argument list containing array-like objects or
        sparse matrices with the same size for their first dimension.
    batch_size : int
        The default batch size.
    shuffle : bool, optional (default=False)
        Whether to shuffle the data before generating batches.
    to_dense : bool, optional (default=True)
        Whether to convert sparse matrices to dense NumPy arrays.

    Raises
    ------
    ValueError
        When no data arrays are provided, or when the input arrays do
        not have the same number of elements.

    """

    def __init__(self, *arrays, batch_size, shuffle=False, to_dense=True):
        self.arrays = arrays
        if self.n_arrays == 0:
            raise ValueError('At least one data array is required')

        for array in arrays[1:]:
            if _num_samples(array) != self.n_samples:
                raise ValueError(
                    'Input arrays must have the same number of elements'
                )

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.to_dense = to_dense
        self.reset()

    def __iter__(self):
        return self

    def __next__(self, batch_size=None):
        """Generate and return a mini-batch

        Parameters
        ----------
        batch_size : int or None, optional (default=None)
            Maximum number of samples in a batch. If not provided, uses
            the default batch size.

        Returns
        -------
        batch:
            An array, or a tuple of arrays (when multiple data arrays
            are provided) containing one mini-batch of data.

        """
        def slice(array, start, end):
            if isinstance(array, sp.spmatrix) and self.to_dense:
                return array[start:end].A
            else:
                return array[start:end]

        if self._pointer == self.n_samples:
            raise StopIteration

        batch_size = batch_size or self.batch_size
        next_pointer = min(self._pointer + batch_size, self.n_samples)
        if self.n_arrays == 1:
            batch = slice(self.arrays[0], self._pointer, next_pointer)
        else:
            batch = (
                slice(array, self._pointer, next_pointer) for array in arrays
            )

        self._pointer = next_pointer
        return batch

    @property
    def n_arrays(self):
        """int: Number of arrays in the dataset"""
        return len(self.arrays)

    @property
    def n_samples(self):
        """int: Number of samples in the dataset"""
        return _num_samples(self.arrays[0])

    def reset(self):
        """Reset the dataset and re-shuffle the data, if necessary"""
        self._pointer = 0
        if self.shuffle:
            if self.n_arrays == 1:
                self.arrays = (sklearn.utils.shuffle(*self.arrays),)
            else:
                self.arrays = sklearn.utils.shuffle(*self.arrays)
