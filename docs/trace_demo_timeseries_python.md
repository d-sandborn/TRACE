# Time series

**Coming soon!**

Estimating the history of C<sub>anth</sub> over time is straightforward with TRACE, but a couple tricks make it much, much faster, which is important when TRACE estimates are desired beyond individual points and casts. 

One of the most significant slowdowns in the C<sub>anth</sub> estimation is the intermediate step of neutral network estimation of preformed properties (TA/Si/N/P) at each point in space. Because these properties are assumed to be time-invariant, this results in the same estimate being made repeatably for ever point in time. The solution is to feed TRACE the preformed properties determined during the first step on all succeeding steps. At the moment, this is implemented as a user-accessible input, requiring the user to run the time series loop once, then again for all other time steps. 

## A single point over multiple times

Time series generation is demonstrated first for a single point for clarity. Initially the "slow" way, involving unnecessary repeated determination of preformed properties.

### The slow way

```python

```

### The fast way

Now a loop is implemented to re-use preformed properties for each location in space for all time steps. 

```python

```

## Many points over many times

The time saved for a single point estimation is minimal, but when TRACE is scaled over many points in space and time, the time savings may add up. 

### The slow way

```python

```
### The fast way

```python

```