# kind <> arithmetic
MEDIAN__MAD = 0  # symmetrical
MEDIAN__MAD_SPLIT_FULL = 8  # asymmetrical

MEDIAN__MAD_SPLIT_PART = 6  # no use

MEDIAN__IQR = 97  # no use
MEDIAN__IQR_SPLIT = 98  # no use

ROB_MEAN__ROB_VAR = 1  # no use
ROB_MEAN__MAD = 2  # no use
ROB_MEAN__MAD_SPLIT_PART = 7  # no use

MEAN__STD = 99  # no use


class ARITHMETIC:
    MEDIAN__MAD = 0  # symmetrical
    MEDIAN__MAD_SPLIT_FULL = 8  # asymmetrical

    MEDIAN__MAD_SPLIT_PART = 6  # no use

    MEDIAN__IQR = 97  # no use
    MEDIAN__IQR_SPLIT = 98  # no use

    ROB_MEAN__ROB_VAR = 1  # no use
    ROB_MEAN__MAD = 2  # no use
    ROB_MEAN__MAD_SPLIT_PART = 7  # no use

    MEAN__STD = 99  # no use


class MEAN_CALC:
    MEDIAN = 'median'
    ROB_MEAN = 'rob_mean'
    MEAN = 'mean'


class DEV_CALC:
    MAD = 'mad'
    MAD_SPLIT_PART = 'part_split_mad'
    MAD_SPLIT_FULL = 'full_split_mad'
    IQR = 'iqr'
    IQR_SPLIT = 'split_iqr'
    STD = 'std'
    ROB_VAR = 'rob_var'
