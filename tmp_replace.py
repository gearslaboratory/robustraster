import sys

try:
    path1 = r'c:\Users\Adriano Matos\Documents\Python Scripts\big-raster-helper\demos\ndvi_us_demo_r_example.ipynb'
    content = open(path1, 'r', encoding='utf-8').read()
    content = content.replace('dask_use_docker', 'use_docker')
    open(path1, 'w', encoding='utf-8').write(content)
    print("Updated " + path1)

    path2 = r'c:\Users\Adriano Matos\Documents\Python Scripts\big-raster-helper\demos\ndvi_us_demo.ipynb'
    content = open(path2, 'r', encoding='utf-8').read()
    content = content.replace('dask_use_docker', 'use_docker')
    open(path2, 'w', encoding='utf-8').write(content)
    print("Updated " + path2)

except Exception as e:
    print(f"FAILED: {e}")
