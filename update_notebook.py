import json

def create_r_demo_notebook():
    file_path = 'demos/ndvi_us_demo.ipynb'
    out_path = 'demos/ndvi_us_demo_r_example.ipynb'
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print("Loaded original notebook.")
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    markdown_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Using R code instead of Python\n",
            "\n",
            "You can also run your function via R code on the Docker workers! Instead of a Python function, you define your function as an R script string."
        ]
    }

    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "r_code = \"\"\"\\\n",
            "compute_ndvi_r <- function(df) {\n",
            "    df$ndvi <- (df$SR_B5 - df$SR_B4) / (df$SR_B5 + df$SR_B4)\n",
            "    return(df[, c(\"time\", \"X\", \"Y\", \"ndvi\")])\n",
            "}\n",
            "\"\"\"\n",
            "\n",
            "df_list = [\"ndvi\"]\n",
            "chunks = {\"time\": 1, \"X\": 2048, \"Y\": 2048}\n",
            "\n",
            "run(\n",
            "    dataset=ic,\n",
            "    source=\"ee\",\n",
            "    preview_dataset = False,\n",
            "    tune_function = False,\n",
            "    dataset_config={\n",
            "        'geometry': Park_Lane_Boundaries,\n",
            "        'crs': 'EPSG:3310',\n",
            "        'scale': 30,\n",
            "    },\n",
            "    user_function_config={\n",
            "        \"is_r_function\": True,\n",
            "        \"r_function_code\": r_code,\n",
            "        \"r_function_name\": \"compute_ndvi_r\",\n",
            "    },\n",
            "    function_tuning_config={\n",
            "        \"chunks\": chunks,\n",
            "        \"max_iterations\": None,\n",
            "        \"output_template\": df_list\n",
            "    },\n",
            "    export_config={\n",
            "        \"mode\": \"raster\",\n",
            "        \"file_format\": \"GTiff\",\n",
            "        \"output_folder\": \"PL_Tuned_NDVI_Tiles_30m_40MIL_R\",\n",
            "        \"vrt\": True,\n",
            "        \"report\": True\n",
            "    },\n",
            "    dask_mode=\"custom\",\n",
            "    dask_config={\n",
            "        \"n_workers\": 4,\n",
            "        \"threads_per_worker\": 1,\n",
            "        \"memory_limit\": \"3g\",\n",
            "    },\n",

            "    docker_image=\"adrianomdocker/rr042\"\n",
            ")"
        ]
    }

    data['cells'].extend([markdown_cell, code_cell])

    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
            print(f"Successfully created {out_path}.")
    except Exception as e:
        print(f"Failed to write: {e}")

if __name__ == "__main__":
    create_r_demo_notebook()
