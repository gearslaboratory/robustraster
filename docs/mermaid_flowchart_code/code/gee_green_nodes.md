flowchart TB

    %% ── TOP LEVEL ───────────────────────────────
    ARD["Analysis Ready Dataset"] --> ADV["Advanced Computing"]

    %% ── LEFT PIPELINE ──────────────────────────
    subgraph PIPELINE_L[" "]
        direction TB

        IN1["Single Image Input"]
        GEE1["GEE Object"]

        UDF1["✖ User-Defined Function"]
        SYS1["System-Defined Function"]

        GEE1_OUT["GEE Object"]
        OUT1["Single Image Output"]

        IN1 --> GEE1

        GEE1 --> UDF1
        GEE1 --> SYS1

        UDF1 --> GEE1_OUT
        SYS1 --> GEE1_OUT

        GEE1_OUT --> OUT1
    end

    %% ── RIGHT PIPELINE ─────────────────────────
    subgraph PIPELINE_R[" "]
        direction TB

        IN2["Single Image Input"]
        GEE2["GEE Object"]

        UDF2["✖ User-Defined Function"]
        SYS2["System-Defined Function"]

        GEE2_OUT["GEE Object"]
        OUT2["Single Image Output"]

        IN2 --> GEE2

        GEE2 --> UDF2
        GEE2 --> SYS2

        UDF2 --> GEE2_OUT
        SYS2 --> GEE2_OUT

        GEE2_OUT --> OUT2
    end

    %% ── CONNECTIONS ────────────────────────────
    ADV --> IN1
    ADV --> IN2

    OUT1 --> TILE["Tiled Dataset"]
    OUT2 --> TILE

    %% ── STYLING ────────────────────────────────
    classDef default fill:#1e1e1e,stroke:#cccccc,color:#ffffff,stroke-width:1.5px;
    classDef green fill:#1e1e1e,stroke:#00c853,color:#b9f6ca,stroke-width:2px;
    classDef red fill:#1e1e1e,stroke:#ff5252,color:#ff8a80,stroke-width:2px;

    class ARD,ADV,IN1,SYS1,OUT1,IN2,SYS2,OUT2 default;
    class GEE1,GEE1_OUT,GEE2,GEE2_OUT,TILE green;
    class UDF1,UDF2 red;