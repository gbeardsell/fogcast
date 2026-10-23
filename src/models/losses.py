"""Masked fine BCE and stable area-pooled native-footprint BCE.

All probability pooling and losses use float32, including under model autocast.
Statistics accumulate sums and counts so a partition is never scored by taking
an unweighted average of batch means. Unsupported metrics are reported as None.
"""