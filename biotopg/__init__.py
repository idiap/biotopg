# Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: MIT

import warnings

from colorama import Fore, Style

from .biotopg import Biotopg

__all__ = ["Biotopg"]

message = (
    f"{Fore.YELLOW}⚠️  biotopg is a work in progress — breaking changes are expected!"
    f"{Style.RESET_ALL}"
)

warnings.warn(message, UserWarning)

__version__ = "0.1.0"
