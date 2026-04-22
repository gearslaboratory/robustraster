flowchart TB

    %% ── TOP LEVEL ───────────────────────────────
    ARD["Analysis Ready Dataset"] --> DASK["Dask"]

    %% ── LEFT PIPELINE ──────────────────────────
    subgraph PIPELINE_L[" "]
        direction TB

        IN1["Single Image Input"]
        XR1["xarray"]

        UDF1["User-Defined Function"]
        SYS1["System-Defined Function"]

        XR1_OUT["xarray"]
        OUT1["Single Image Output"]

        IN1 --> XR1

        XR1 --> UDF1
        XR1 --> SYS1

        UDF1 --> XR1_OUT
        SYS1 --> XR1_OUT

        XR1_OUT --> OUT1
    end

    %% ── RIGHT PIPELINE ─────────────────────────
    subgraph PIPELINE_R[" "]
        direction TB

        IN2["Single Image Input"]
        XR2["xarray"]

        UDF2["User-Defined Function"]
        SYS2["System-Defined Function"]

        XR2_OUT["xarray"]
        OUT2["Single Image Output"]

        IN2 --> XR2

        XR2 --> UDF2
        XR2 --> SYS2

        UDF2 --> XR2_OUT
        SYS2 --> XR2_OUT

        XR2_OUT --> OUT2
    end

    %% ── CONNECTIONS ────────────────────────────
    DASK --> IN1
    DASK --> IN2

    OUT1 --> MOSAIC["Mosaicked Dataset"]
    OUT2 --> MOSAIC

    %% ── STYLING ────────────────────────────────
    classDef default fill:#1e1e1e,stroke:#cccccc,color:#ffffff,stroke-width:1.5px;
    classDef infra fill:#1e1e1e,stroke:#00c853,color:#b9f6ca,stroke-width:2px;

    class ARD,IN1,UDF1,SYS1,OUT1,IN2,UDF2,SYS2,OUT2,MOSAIC default;
    class DASK,XR1,XR1_OUT,XR2,XR2_OUT infra;