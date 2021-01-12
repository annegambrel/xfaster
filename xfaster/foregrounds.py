from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# Dictionary containing foreground template look-up tables.
# Templates are made out of processed Planck maps.
# Updated: 2017-03-08
# ----------------------

foreground_codes = {
    1: "lfi44-lfi30",
    2: "lfi70-lfi44",
    3: "hfi100-lfi70",
    5: "hfi217-hfi143",
    6: "hfi353-hfi217",
    13: "hfi353-hfi100",
    15: "hfi353-spider90",
    16: "hfi353-hfi143",
    18: "lfi70-lfi30",
}

foreground_templates = list(foreground_codes.keys())
foreground_names = list(foreground_codes.values())
foreground_codes_inv = {foreground_codes[k]: k for k in foreground_codes}
