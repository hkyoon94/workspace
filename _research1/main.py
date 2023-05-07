import os
import sys

sys.path.append(os.path.abspath("/home/honggyu/workspace"))

from test_pkg.pkg1.module import add
from test_pkg.pkg2.module import mul

add(2, 3)
mul(3, 4)
add(3, 6)
