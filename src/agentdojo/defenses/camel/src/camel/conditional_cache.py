# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from functools import lru_cache, wraps


def conditional_lru_cache(maxsize=None, typed=False):
    """
    Decorator to conditionally apply lru_cache based on argument hashability.

    Args:
        maxsize (int, optional):  Maximum size of the cache. Defaults to 128.
        typed (bool, optional): Cache different argument types separately if True. Defaults to False.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            all_args_hashable = True
            try:
                for arg in args:
                    hash(arg)  # Try hashing each positional argument
                for value in kwargs.values():
                    hash(value)  # Try hashing each keyword argument value
            except TypeError:
                all_args_hashable = False

            if all_args_hashable:
                # Arguments are hashable, use lru_cache
                @lru_cache(maxsize=maxsize, typed=typed)
                def cached_func(*args_cache, **kwargs_cache):  # Different name to avoid shadowing
                    return func(*args_cache, **kwargs_cache)

                return cached_func(*args, **kwargs)  # Call the cached version
            else:
                # Arguments are not hashable, just compute
                return func(*args, **kwargs)

        return wrapper

    return decorator
