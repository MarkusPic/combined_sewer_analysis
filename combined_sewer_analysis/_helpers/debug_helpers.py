from functools import wraps

fn_middle = u'\u250c' + u'\u2500' * 2  # '├── '
fn_last = u'\u2514' + u'\u2500' * 2  # '└── '
par_middle = '   '
par_last = u'\u2502' + '  '  # '│   '


try:
    raise ModuleNotFoundError
    from mp.helpers import class_timeit as timeit, check
    from mp.helpers.check_time import lev
except ModuleNotFoundError:
    def timeit(method):
        @wraps(method)
        def timed(*args, **kwargs):
            return method(*args, **kwargs)

        return timed


    def check(*args):
        return


    lev = ''
