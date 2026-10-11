# Combined Sewer Analysis

This is a companion repository for the research article "Decomposition of Dry- and Wet-Weather Flows in Combined Sewer Measurements" by Pichler & Muschalla (2026).

Please cite as:
> Pichler, M., & Muschalla, D. (2026). Decomposition of Dry- and Wet-Weather Flows in Combined Sewer Measurements. Water, 18(20), 2486. https://doi.org/10.3390/w18202486


```python
import pandas as pd
from combined_sewer_analysis import AnalyseData

# long term high resolution flow-rate or load-rate time series of combined sewer observations.
ts = pd.Series(index=pd.DatetimeIndex(...), data=...)
csa = AnalyseData(ts)

fig, ax = csa.get_interim_figure(slice('2009-07-01', '2009-07-06'))
fig.savefig("example_figure.png")
```

![example_figure.png](example_figure.png)