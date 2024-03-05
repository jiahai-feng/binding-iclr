def overwrite(__d, /, **kwargs):
    return {**__d, **kwargs}


def chain(preprocessor, func):
    def wrapper(**kwargs):
        newkwargs = preprocessor(**kwargs)
        # newkwargs will silently override kwargs
        return func(**{**kwargs, **newkwargs})

    return wrapper


def terminate(func):
    return lambda preprocessor: chain(preprocessor, func)


def append(postprocessor):
    if isinstance(postprocessor, list):
        if postprocessor:
            return lambda func: append(postprocessor[1:])(
                append(postprocessor[0])(func)
            )
        else:
            return lambda func: func
    else:
        return lambda preprocessor: chain(preprocessor, inject_output()(postprocessor))


def prepend(preprocessor):
    if isinstance(preprocessor, list):
        if preprocessor:
            return lambda func: prepend(preprocessor[:-1])(
                prepend(preprocessor[-1])(func)
            )
        else:
            return lambda func: func
    else:
        return lambda func: chain(preprocessor, func)


def inject_output(output_name=None):
    def outer(func):
        if output_name is None:

            def wrapper(**kwargs):
                return {**kwargs, **func(**kwargs)}

            return wrapper
        else:

            def wrapper(**kwargs):
                return {**kwargs, **{output_name: func(**kwargs)}}

            return wrapper

    return outer


def set_defaults(**default_kwargs):
    def outer(func):
        def wrapper(**kwargs):
            return func(**{**default_kwargs, **kwargs})

        return wrapper

    return outer


def apply_decorators(decorators, func):
    """
    Applies decorators in reverse order, according to python convention
    """
    for decorator in decorators[::-1]:
        func = decorator(func)
    return func


"""
Caching utils
"""
import unicodedata
import re
import os
import torch


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def get_cache_path(cache_path, *args):
    return os.path.join(cache_path, *[slugify(arg) for arg in args])


def cache(name, thing, prefixes, force_compute=False, read_only=False, verbose=True):
    file_path = get_cache_path(*prefixes, name) + ".pt"
    if not force_compute and os.path.exists(file_path):
        if verbose:
            print(f"using cached {file_path}")
        return torch.load(file_path)
    else:
        if read_only:
            raise Exception("{file_path} does not exist")
        if not os.path.exists(os.path.dirname(file_path)):
            if verbose:
                print(f"making cache path {os.path.dirname(file_path)}")
            os.makedirs(os.path.dirname(file_path))
        result = thing()
        torch.save(result, file_path)
        return result
