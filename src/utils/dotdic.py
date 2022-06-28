from copy import deepcopy
from types import SimpleNamespace


class DotDic(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        return DotDic(deepcopy(dict(self), memo=memo))

    def as_namespace(self) -> SimpleNamespace:
        """This function returns the dictionary as a namespace."""
        return SimpleNamespace(**self)
