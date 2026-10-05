# Generated artifacts

`fraudlens export` writes the compact JSON in this directory. It is committed on
purpose: the 697 MB Elliptic dataset cannot be, so these files are what let a fresh
clone — and the API in artifact mode — show real results without any download.

| File | Contents |
|---|---|
| `metrics.json` | the single source of truth for every reported number |
| `overview.json` | dataset stats, score distribution, timeline, model comparison |
| `graph_sample.json` | connected, fraud-rich node sample plus induced edges |
| `rings.json` | top fraud rings with members and induced subgraphs |
| `explanations.json` | precomputed explanations for ~20 notable nodes |
| `nodes_index.json` | searchable node subset with 1-hop neighbours |
| `features.json` | feature index → name and local/aggregated group |

Regenerate with:

```bash
fraudlens export
```
