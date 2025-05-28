# Function Tuning with `tune_function`

The `tune_function` parameter in `run()` is designed to automatically optimize how your custom function is applied to large datasets. It evaluates your dataset and available system resources to determine the most efficient way to process the data in parallel.

---

## What Does It Do?

If you set `tune_function` to `True`, it will take in two things:

1. Your **dataset object**
2. Your **custom function**

It then runs a benchmarking routine to tune the function execution based on how your system performs.

---

## How Does It Work?

Under the hood, this method leverages **xarray chunking**, which breaks large datasets into smaller pieces that can be processed independently and in parallel. Xarray supports lazy, out-of-core computation via Dask, and chunking is the key to enabling:

- Memory-efficient processing
- Parallel execution
- Disk-based storage for large-scale workflows

### Why Chunking Matters

Rather than loading and processing an entire dataset at once (which often won’t fit in memory), chunking processes one manageable section at a time. These chunks are:

- Loaded into memory
- Processed with your custom function
- Discarded (or written to disk), freeing up space for the next batch of chunks

Choosing the **right chunk size** is critical. Too small? You waste overhead managing chunks. Too large? You run out of memory.

---

## What Does `tune_function` Actually Do?

The tuning process follows this logic:

1. **Start with the smallest valid chunk**.
2. Run the user function on a single chunk.
3. Record performance metrics (such as time and memory usage).
4. Increase the chunk size by a factor of 2.
5. At each iteration, compare the most recent compute time to the previous one.
   - If performance worsens (i.e., compute time increases), stop and return the chunk size from the previous iteration.
   - If performance is better (i.e. compute time decreases), repeat steps 2-4.
7. Use the best-performing chunk size to do a full run of your function on the dataset.

The tuning process **stops early** if the system runs out of memory or if performance degrades.

---

## What Does It Output?

Nothing really. You won't see anything happen on your screen. The best-performing chunk size is used to split apart your dataset into chunks of this size and then the full computation begins.

## Last Note

`tune_function` is entirely optional. If set to `False`, tuning is skipped entirely and a default chunk size is used. The default chunk size will already do a pretty good job spliting apart your data in a way that will be effcient for your needs. At the moment, `tune_function` adds the time it takes to download data from Google Earth Engine to your machine as part of the computation process, which is not optimal. In the future, I would like to update my tuning to remove this as part of the optimization process as network speeds flucuate from user to user (additionally, Earth Engine is a free resource and can vary in it's download speeds as well.) You are also welcome to set `tune_function` to `False` and pass in your own custom chunk size in `export_kwargs` uisng the keyword `chunks`.