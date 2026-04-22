flowchart TB

    IN["Single Image Input"]
    DF1["DataFrame"]

    UDF["User Defined Function"]
    SYS["✖ System Defined Function"]

    DF2["DataFrame"]
    OUT["Single Image Output"]

    IN --> DF1

    DF1 --> UDF
    DF1 --> SYS

    UDF --> DF2
    SYS --> DF2

    DF2 --> OUT

    %% Styling
    classDef active fill:#1e1e1e,stroke:#00c853,color:#e8f5e9,stroke-width:2px;
    classDef disabled fill:#1e1e1e,stroke:#ff5252,color:#ff8a80,stroke-width:2px;

    class IN,DF1,UDF,DF2,OUT active;
    class SYS disabled;