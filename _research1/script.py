import os
import sys

from test_pkg.pkg1.module import add
from test_pkg.pkg2.module import mul

module_path = os.path.abspath(os.path.join(".."))

if module_path not in sys.path:
    sys.path.append(module_path)

add(2, 3)
mul(3, 4)
add(3, 6)
