class ARITHMETIC:
    MEDIAN__MAD = 0  # symmetrical
    MEDIAN__MAD_SPLIT_FULL = 8  # asymmetrical

    MEDIAN__MAD_SPLIT_PART = 6  # used in TEMPEST R05 | asymmetrical between 7 and 12 (morning peak)

    MEDIAN__IQR = 97  # no use
    MEDIAN__IQR_SPLIT = 98  # no use

    ROB_MEAN__ROB_VAR = 1  # no use
    ROB_MEAN__MAD = 2  # no use
    ROB_MEAN__MAD_SPLIT_PART = 7  # used in master thesis

    MEAN__STD = 99  # no use

    list_of_median = (MEDIAN__MAD, MEDIAN__MAD_SPLIT_FULL, MEDIAN__MAD_SPLIT_PART, MEDIAN__IQR, MEDIAN__IQR_SPLIT)
    list_of_robust_mean = (ROB_MEAN__ROB_VAR, ROB_MEAN__MAD, ROB_MEAN__MAD_SPLIT_PART)
    list_of_mean = (MEAN__STD, )

    list_of_mad = (MEDIAN__MAD, MEDIAN__MAD_SPLIT_FULL, MEDIAN__MAD_SPLIT_PART, ROB_MEAN__MAD, ROB_MEAN__MAD_SPLIT_PART)


# class MEAN_CALC:
#     MEDIAN = 'median'
#     ROB_MEAN = 'rob_mean'
#     MEAN = 'mean'
#
#
# class DEV_CALC:
#     MAD = 'mad'
#     MAD_SPLIT_PART = 'part_split_mad'
#     MAD_SPLIT_FULL = 'full_split_mad'
#     IQR = 'iqr'
#     IQR_SPLIT = 'split_iqr'
#     STD = 'std'
#     ROB_VAR = 'rob_var'

MAD_TO_STD = 0.6745  # mad/std
STD_TO_MAD = 1 / MAD_TO_STD  # std = mad * 1.4826

# 2 x σ = 2.965 x MAD ... -> 95%
