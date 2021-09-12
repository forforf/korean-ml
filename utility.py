# credit: https://stackoverflow.com/posts/62420097/revisions
from collections.abc import Iterable

BASE_TYPES = [str, int, float, bool, type(None)]

def base_typed(obj):
    """Recursive reflection method to convert any object property into a comparable form.
    """
    T = type(obj)
    from_numpy = T.__module__ == 'numpy'

    if T in BASE_TYPES or callable(obj) or (from_numpy and not isinstance(T, Iterable)):
        return obj

    if isinstance(obj, Iterable):
        base_items = [base_typed(item) for item in obj]
        # noinspection PyCallingNonCallable
        return base_items if from_numpy else T(base_items)

    d = obj if T is dict else obj.__dict__

    return {k: base_typed(v) for k, v in d.items()}


def deep_equals(*args):
    return all(base_typed(args[0]) == base_typed(other) for other in args[1:])