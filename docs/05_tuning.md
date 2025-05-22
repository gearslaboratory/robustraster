# Function Tuning with `tune_function`

The `tune_function` parameter in `run()` is designed to automatically optimize how your custom function is applied to large datasets. It evaluates your dataset and available system resources to determine the most efficient way to process the data in parallel.

---

## What Does It Do?

If you set `tune_function` to `True`, it will take in two things:

1. Your **dataset object** (an `xarray.Dataset`)
2. Your **custom function** (a callable that operates on a chunk of data)

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
- Discarded (or written to disk), freeing up space for the next batch

Choosing the **right chunk size** is critical. Too small? You waste overhead managing chunks. Too large? You run out of memory.

---

## What Does `tune_function` Actually Do?

The tuning process follows this logic:

1. **Start with the smallest valid chunk**.
2. Run the user function on a single chunk.
3. Record performance metrics (such as time and memory usage).
4. Increase the chunk size by a factor of 2.
5. Repeat steps 2–4.
6. At each iteration, compare the most recent compute time to the previous one.
   - If performance worsens (i.e., time increases), return the chunk size from the previous iteration.
7. Use the best-performing chunk size to do a full run of your function on the dataset.

The tuning process **stops early** if the system runs out of memory or if performance degrades.

---

## What Does It Output?

- A **JSON file** containing the optimal chunk configuration.
- This file can later be reused when applying your function (you don’t have to re-tune it).

```json
{
  "chunks": {
    "time": 64,
    "X": 512,
    "Y": 256
  },
  "compute_time": 3.41,
  "timestamp": "2025-05-10T22:14:03"
}
