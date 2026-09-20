### Table: Aggregate Empirical Metrics Across 324 Experimental Conditions

| Model Scale | Classification Regime | Median Latency ($p50$) | Grounding Recall (%) | Output Tokens | Token Efficiency ($\eta$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Small Local (Qwen 1.7B)** | Control | 905.2 ms | 83.1% | 137.5 tok | **0.647** |
| **Small Local (Qwen 1.7B)** | Regex Trie | 567.5 ms | 60.7% | 83.7 tok | **0.721** |
| **Small Local (Qwen 1.7B)** | TypeSafe Jev | 576.2 ms | 65.7% | 85.7 tok | **0.754** |
| **Medium Local (Qwen 7B 4-bit)** | Control | 1711.6 ms | 63.1% | 86.5 tok | **0.804** |
| **Medium Local (Qwen 7B 4-bit)** | Regex Trie | 1534.5 ms | 63.8% | 76.0 tok | **0.874** |
| **Medium Local (Qwen 7B 4-bit)** | TypeSafe Jev | 1512.4 ms | 62.4% | 72.1 tok | **0.930** |
| **Frontier Cloud (Gemini 3.8 Flash)** | Control | 2128.3 ms | 73.5% | 105.6 tok | **0.722** |
| **Frontier Cloud (Gemini 3.8 Flash)** | Regex Trie | 2028.4 ms | 70.4% | 109.1 tok | **0.590** |
| **Frontier Cloud (Gemini 3.8 Flash)** | TypeSafe Jev | 2097.9 ms | 73.2% | 108.3 tok | **0.691** |

### Table: Domain Breakdown Performance (4 Enterprise Telco Domains)

| Domain | Model Scale | Control Latency | Regex Latency | Jev Latency | Jev Latency Delta | Control Recall | Jev Recall | Jev Efficiency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fiber | Small Local | 915.9 ms | 528.2 ms | **532.0 ms** | -383.8 ms (+41.9%) | 78.7% | 48.8% | **0.616** |
| Fiber | Medium Local | 1746.7 ms | 1513.5 ms | **1601.7 ms** | -145.1 ms (+8.3%) | 45.2% | 56.4% | **0.733** |
| Fiber | Frontier Cloud | 2139.4 ms | 1985.3 ms | **2199.5 ms** | +60.1 ms (-2.8%) | 68.2% | 64.3% | **0.585** |
| Billing | Small Local | 744.3 ms | 566.9 ms | **594.2 ms** | -150.1 ms (+20.2%) | 82.9% | 77.5% | **0.881** |
| Billing | Medium Local | 1271.4 ms | 1380.5 ms | **1379.5 ms** | +108.1 ms (-8.5%) | 67.9% | 65.2% | **1.034** |
| Billing | Frontier Cloud | 2390.7 ms | 1812.0 ms | **1964.9 ms** | -425.8 ms (+17.8%) | 66.7% | 62.7% | **0.808** |
| Network | Small Local | 996.4 ms | 558.7 ms | **526.4 ms** | -469.9 ms (+47.2%) | 83.0% | 49.9% | **0.624** |
| Network | Medium Local | 2192.5 ms | 1664.9 ms | **1638.1 ms** | -554.4 ms (+25.3%) | 53.7% | 46.3% | **0.593** |
| Network | Frontier Cloud | 1975.1 ms | 2059.1 ms | **2617.1 ms** | +642.0 ms (-32.5%) | 80.9% | 86.5% | **0.582** |
| Contract | Small Local | 964.2 ms | 616.3 ms | **652.1 ms** | -312.1 ms (+32.4%) | 87.8% | 86.7% | **0.896** |
| Contract | Medium Local | 1635.5 ms | 1579.0 ms | **1430.3 ms** | -205.3 ms (+12.5%) | 85.5% | 81.7% | **1.361** |
| Contract | Frontier Cloud | 2008.0 ms | 2257.3 ms | **1610.0 ms** | -398.0 ms (+19.8%) | 78.3% | 79.4% | **0.791** |