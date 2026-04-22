flowchart TB

    ARD["Analysis Ready Dataset"]
    ADV["Advanced Computing"]
    MOSAIC["Mosaicked Dataset"]

    ARD --> ADV

    subgraph TOP_PIPE[" "]
        direction LR
        IN1["Single Image Input"] --> DF1["DataFrame"]
        DF1 --> UDF1["User-Defined Function"]
        DF1 --> SYS1["System-Defined Function"]
        UDF1 --> DF1O["DataFrame"]
        SYS1 --> DF1O
        DF1O --> OUT1["Single Image Output"]
    end

    subgraph BOTTOM_PIPE[" "]
        direction LR
        IN2["Single Image Input"] --> DF2["DataFrame"]
        DF2 --> UDF2["User-Defined Function"]
        DF2 --> SYS2["System-Defined Function"]
        UDF2 --> DF2O["DataFrame"]
        SYS2 --> DF2O
        DF2O --> OUT2["Single Image Output"]
    end

    ADV --> IN1
    ADV --> IN2

    OUT1 --> MOSAIC
    OUT2 --> MOSAIC