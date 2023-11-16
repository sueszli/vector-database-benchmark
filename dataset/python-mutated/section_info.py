"""
This modules exctracts some parameters for a given section
"""

def get_nonzero_Mmax(bin_mag, cumulative_rates):
    if False:
        while True:
            i = 10
    '\n    gets the maximum magnitude with a non zero rate for this fault.\n\n    bin_mag : list, binning in magnitude for the mfd\n    cumulative_rates : cumulative participation rate for the section\n    '
    Mmax = bin_mag[0]
    for (mag, rate) in zip(bin_mag, cumulative_rates):
        if rate != 0.0:
            Mmax = mag
    return Mmax